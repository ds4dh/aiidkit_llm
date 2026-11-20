import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
import yaml
import json
import shutil
import argparse
import optuna
import threading
import subprocess
import queue
import numpy as np
from scipy.stats import gmean
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
try:
    from optuna.storages.journal import JournalFileBackend as _JournalFileBackend
except ImportError:
    from optuna.storages import JournalFileStorage as _JournalFileBackend
from src.utils import apply_config_overrides
from src.model.model_utils import resolve_best_checkpoint
from src.model.patient_embedder import PatientEmbeddingModelFactory
from scripts.script_utils import format_pretrain_run_id, get_value_embedding_tag


class FatalWorkerError(BaseException):
    """Raised when a worker hits repeated consecutive trial crashes and should abort fast."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage Optuna hyperparameter optimization for t-EAV Transformer (Stage 1: pretrain, Stage 2: finetune)."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Optimization stage: 'pretrain' (self-supervised pre-training) or 'finetune' (supervised multi-task fine-tuning from Stage-1 checkpoint).",
    )
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
    parser.add_argument("--split", type=str, default=None, help="Override data_split_type (default: from config).")
    parser.add_argument(
        "--tasks",
        type=str,
        default="multitask_bacteria,multitask_viruses,multitask_fungi,death,graft_loss",
        help="Comma-separated task key(s) for fine-tuning stage (default: canonical 5-run B_CAT_MT setup).",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=None,
        help="For --stage finetune: path to pretraining folder containing checkpoint-* (default: auto-resolved from Stage-1 best_pretrain_config.json or baseline pretraining).",
    )
    parser.add_argument(
        "--export-run-id",
        type=str,
        default="optuna_best",
        help="Canonical run_id under results/<family>/<split>/<run_id>_<raw|bin> where the winning checkpoint/predictions are linked (default: optuna_best).",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=None,
        help="Global target of COMPLETE trials across all parallel workers (default: optuna.num_trials from config; 0 disables global cap).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Max Optuna trials to execute in this worker process (default: run until --target-trials reached; pass 0 to finalize/inspect study).",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort worker if this many consecutive trials crash (prevents burning budget on config/hardware errors).",
    )
    parser.add_argument(
        "--reap-running",
        action="store_true",
        help="Before starting/finalizing, mark any orphaned RUNNING trials (from dead workers) as FAIL and clean their partial checkpoints.",
    )
    parser.add_argument(
        "--compact-journal",
        action="store_true",
        help="Deduplicate redundant CREATE_STUDY (op_code=0) entries in the journal file so external Optuna viewers show a single clean study.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize the Optuna journal study once (if needed) and exit immediately without running trials or finalization.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip post-study symlink/eval-only finalization (use on parallel Slurm workers when a dedicated --n-trials 0 finalizer runs afterok).",
    )
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Optional JSON or key=value config overrides (e.g. use_numerical_embedding=false).")
    parser.add_argument("--debug", "-d", action="store_true", help="Run a quick debug check with capped steps and trials.")
    return parser.parse_args()


cli_args = parse_args()
DEBUG = cli_args.debug
STAGE = cli_args.stage
CONFIG_PATH = cli_args.config
with open(CONFIG_PATH, "r") as f:
    CLI_CFG = yaml.safe_load(f)
if cli_args.overrides:
    CLI_CFG = apply_config_overrides(CLI_CFG, cli_args.overrides)

SPLIT_TYPE = cli_args.split or CLI_CFG.get("data_split_type", "random_split")
TASKS_ARG = cli_args.tasks
VAL_TAG = get_value_embedding_tag(CLI_CFG)
EXPORT_RUN_ID = f"{cli_args.export_run_id}_{VAL_TAG}" if not cli_args.export_run_id.endswith(f"_{VAL_TAG}") else cli_args.export_run_id
BASE_RESULTS_DIR = Path(CLI_CFG["result_dir"])
IS_CAUSAL = PatientEmbeddingModelFactory._is_causal_backbone(
    CLI_CFG["model"]["model_id"],
    architecture_type=CLI_CFG["model"].get("architecture_type"),
)

# Isolate Optuna directory per model family (results/transformer vs results/generative_transformer)
OPTUNA_DIR = BASE_RESULTS_DIR / f"optuna_{STAGE}" / f"{SPLIT_TYPE}_{VAL_TAG}"
DEFAULT_TOTAL_TRIALS = int(CLI_CFG["optuna"]["num_trials"] if not DEBUG else CLI_CFG["optuna"]["num_trials_debug"])
TARGET_TRIALS = cli_args.target_trials if cli_args.target_trials is not None else DEFAULT_TOTAL_TRIALS
N_TRIALS = cli_args.n_trials  # None = run until TARGET_TRIALS reached globally; 0 = finalize only

STUDY_NAME = f"{CLI_CFG['optuna']['study_name']}_{STAGE}_{SPLIT_TYPE}_{VAL_TAG}"
if STAGE == "finetune":
    safe_task_tag = TASKS_ARG.replace(",", "-")
    STUDY_NAME = f"{STUDY_NAME}_{safe_task_tag}"
    OPTUNA_DIR = OPTUNA_DIR / safe_task_tag

BEST_MODEL_LOCK = threading.Lock()
BEST_MODEL_INFO = {
    "score": -float("inf"),
    "path": None,
    "params": None,
    "overrides": None,
    "portable_overrides": None,
}
FAILURE_STATE = {"consecutive": 0}

if cli_args.init_only or (cli_args.n_trials == 0 and STAGE == "pretrain"):
    GPU_IDS = ["0"]
else:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        GPU_IDS = [g.strip() for g in cuda_visible.split(",") if g.strip()]
    else:
        import torch
        n_dev = torch.cuda.device_count()
        GPU_IDS = [str(i) for i in range(max(1, n_dev))]

NUM_GPUS = len(GPU_IDS)
GPU_QUEUE: queue.Queue[str] = queue.Queue()
for gid in GPU_IDS:
    GPU_QUEUE.put(gid)


def _resolve_scratch_trial_root() -> Path | None:
    """
    Prefer node-local fast scratch (NVMe /scratch/$USER/$SLURM_JOB_ID or container /tmp bind)
    for transient ~1.8 GB trial checkpoints so shared /home (BeeGFS) never fills up during HPO.
    """
    home_resolved = str(Path.home().resolve())
    candidates = []
    env_scratch = os.environ.get("AIIDKIT_SCRATCH_DIR", "").strip()
    if env_scratch:
        candidates.append(Path(env_scratch))
    slurm_jid = os.environ.get("SLURM_JOB_ID", "").strip()
    user = os.environ.get("USER", "").strip()
    if slurm_jid and user:
        candidates.append(Path(f"/scratch/{user}/{slurm_jid}"))
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir:
        candidates.append(Path(tmpdir))

    for cand in candidates:
        try:
            if cand.exists() and os.access(cand, os.W_OK):
                resolved = str(cand.resolve())
                if not resolved.startswith(home_resolved):
                    root = cand / f"optuna_{BASE_RESULTS_DIR.name}_{STAGE}_{SPLIT_TYPE}_{os.getpid()}"
                    root.mkdir(parents=True, exist_ok=True)
                    return root
        except OSError:
            continue
    return None


SCRATCH_TRIAL_ROOT = _resolve_scratch_trial_root()


def compact_journal_duplicate_studies(log_file: str) -> int:
    """
    Remove redundant duplicate `CREATE_STUDY` (op_code == 0) records with the same `study_name`
    written by legacy `optuna.create_study(..., load_if_exists=True)` calls across multi-worker jobs.
    Preserves study_id=0 and all trials bit-for-bit while eliminating empty 'ghost' studies in viewers.
    """
    p = Path(log_file)
    if not p.exists() or p.stat().st_size == 0:
        return 0

    seen_studies: set[str] = set()
    kept_lines: list[str] = []
    removed = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            if rec.get("op_code") == 0:
                sname = rec.get("study_name", "")
                if sname in seen_studies:
                    removed += 1
                    continue
                seen_studies.add(sname)
            kept_lines.append(line if line.endswith("\n") else line + "\n")

    if removed > 0:
        tmp_p = p.with_suffix(p.suffix + f".compact_{os.getpid()}.tmp")
        with open(tmp_p, "w", encoding="utf-8") as out:
            out.writelines(kept_lines)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_p, p)
        print(f"[Journal] Compacted {removed} duplicate CREATE_STUDY record(s) from {p}")
    return removed


def load_or_create_study_once(log_file: str, study_name: str, sampler: TPESampler) -> tuple[optuna.Study, JournalStorage]:
    """
    Attach to an existing JournalStorage study via `optuna.load_study` FIRST (0 writes to journal!),
    and only acquire an atomic cross-node symlink lock to call `optuna.create_study` once if missing.
    Prevents N parallel Slurm workers from appending N duplicate op_code=0 ('ghost study') entries.
    """
    storage = JournalStorage(_JournalFileBackend(log_file))
    try:
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        return study, storage
    except KeyError:
        pass

    lock_path = f"{log_file}.init.lock"
    deadline = time.monotonic() + 60.0
    acquired = False
    while time.monotonic() < deadline:
        try:
            os.symlink(f"{os.uname().nodename}:{os.getpid()}", lock_path)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.1)

    try:
        # Re-open storage to sync any study created while we waited for the lock
        storage = JournalStorage(_JournalFileBackend(log_file))
        try:
            study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
            return study, storage
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=False,
                sampler=sampler,
            )
            # Enqueue the canonical baseline configuration as Trial #0 on a fresh pretraining study
            # so TPE starts with an exact apples-to-apples anchor point!
            if STAGE == "pretrain":
                base_anchor = {
                    "mask_ent": 0.00,
                    "mask_att": 0.15,
                    "mask_val": 0.60,
                    "lr_pre": 1.0e-4,
                    "weight_decay_pre": 1.0e-2,
                    "lr_scheduler_pre": "cosine",
                    "warmup_base_pre": 100,
                    "grad_accum_pre": 4,
                    "dropout_embedding": 0.00,
                    "dropout_attention": 0.10,
                }
                if not IS_CAUSAL:
                    base_anchor.update({
                        "local_attention": 128,
                        "dropout_mlp": 0.10,
                    })
                study.enqueue_trial(base_anchor)
            return study, storage
    finally:
        if acquired:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass


def reap_stale_running_trials(study: optuna.Study, storage: JournalStorage) -> int:
    """Mark orphaned RUNNING trials (left behind by killed/crashed workers) as FAIL and clean partial checkpoints."""
    reaped = 0
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        cleanup_checkpoints(OPTUNA_DIR / f"trial_{t.number:03d}")
        reaped += 1
    if reaped > 0:
        print(f"[Reaper] Marked {reaped} orphaned RUNNING trial(s) as FAIL and cleaned partial checkpoints.")
    return reaped


def get_run_id(masking_rules: dict) -> str:
    return format_pretrain_run_id(CLI_CFG, masking_rules)


def resolve_source_pretrained_dir() -> tuple[Path, dict | None]:
    """Locate the pretrained checkpoint directory (and optional Stage-1 metadata) to seed Stage 2 trials."""
    if cli_args.pretrained_dir:
        p = Path(cli_args.pretrained_dir)
        if p.exists():
            return p, None
        raise FileNotFoundError(f"Specified --pretrained-dir does not exist: {p}")

    # Check family-isolated Stage-1 config first, then legacy shared location
    for stage1_meta in (
        BASE_RESULTS_DIR / "optuna_pretrain" / f"{SPLIT_TYPE}_{VAL_TAG}" / "best_pretrain_config.json",
        BASE_RESULTS_DIR.parent / "optuna_pretrain" / f"{SPLIT_TYPE}_{VAL_TAG}" / "best_pretrain_config.json",
    ):
        if stage1_meta.exists():
            with open(stage1_meta, "r") as f:
                meta = json.load(f)
            best_pt = meta.get("best_pretraining_dir")
            if best_pt and Path(best_pt).exists():
                return Path(best_pt), meta

    # Fallback to canonical optuna_best or default pretraining directory
    for cand_pt in (
        BASE_RESULTS_DIR / SPLIT_TYPE / EXPORT_RUN_ID / "pretraining",
        BASE_RESULTS_DIR / SPLIT_TYPE / get_run_id(CLI_CFG["data_collator"]["mlm_masking_rules"]) / "pretraining",
    ):
        if cand_pt.exists():
            return cand_pt, None
    raise FileNotFoundError(
        f"Could not find a pretrained directory under {BASE_RESULTS_DIR / SPLIT_TYPE}. Run Stage-1 pretraining first or pass --pretrained-dir."
    )


def run_subprocess(
    script_path: str,
    overrides: dict,
    env: dict,
    extra_args: list[str] | None = None,
    error_log_path: Path | None = None,
):
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        CONFIG_PATH,
    ]
    for ov in (cli_args.overrides or []):
        cmd.extend(["--overrides", ov])
    cmd.extend(["--overrides", json.dumps(overrides), "--silent"])
    if extra_args:
        cmd.extend(extra_args)

    child_env = env.copy()
    if not DEBUG:
        # Prevent child Trainer progress bars / 400-metric eval dicts from bloating worker AIIDKIT_LOG_FILE
        child_env.pop("AIIDKIT_LOG_FILE", None)
        result = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    else:
        result = subprocess.run(cmd, env=child_env)

    if result.returncode != 0:
        gpu_id = env.get("CUDA_VISIBLE_DEVICES", "Unknown")
        tail_err = (result.stderr or "")[-3000:] if not DEBUG else ""
        tail_out = (result.stdout or "")[-1500:] if not DEBUG else ""
        print(f"\n[!] Error running {script_path} on GPU {gpu_id} (exit code {result.returncode})")
        if tail_err:
            print(tail_err)
        elif tail_out:
            print(tail_out)
        if error_log_path is not None:
            try:
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                error_log_path.write_text(
                    f"Command: {' '.join(cmd)}\nExit code: {result.returncode}\n\n--- STDOUT TAIL ---\n{tail_out}\n\n--- STDERR TAIL ---\n{tail_err}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        raise RuntimeError(f"Script {script_path} failed with exit code {result.returncode}.")


def evaluate_pretrain_trial(pretrain_dir: Path) -> float:
    """
    Read standardized validation metrics from `pretraining_results.json`
    (evaluated on the fixed reference mask e00-a15-v60 with seed 1234).
    """
    res_file = pretrain_dir / "pretraining_results.json"
    if res_file.exists():
        with open(res_file, "r") as f:
            data = json.load(f)
        acc = float(data.get("ref_eval_mlm_accuracy", 0.0))
        sil = float(data.get("ref_eval_silhouette_score", -1.0))
        bonus = 0.02 * max(0.0, sil) if sil > 0 else 0.0
        return acc + bonus

    try:
        best_ckpt = resolve_best_checkpoint(pretrain_dir)
    except Exception:
        best_ckpt = None
    if best_ckpt is not None:
        ts_path = best_ckpt / "trainer_state.json"
        if ts_path.exists():
            with open(ts_path, "r") as f:
                ts = json.load(f)
            if ts.get("best_metric") is not None:
                return float(ts["best_metric"])
    return 0.0


def evaluate_finetune_trial(run_path: Path) -> float:
    """
    Aggregate validation ROC-AUC scores from `validation_results.json` using a
    log-support-weighted geometric mean across all valid heads (strictly on validation split).
    Weighting by log(1 + N_pos_val) prevents ultra-rare 30d heads (~4 validation positives)
    from dominating trial ranking via sampling noise while keeping every pathogen group represented.
    """
    finetuning_path = run_path / "finetuning"
    if not finetuning_path.exists():
        print(f"Path not found: {finetuning_path}")
        return 0.0

    scores = []
    weights = []
    for result_file in sorted(finetuning_path.rglob("validation_results.json")):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            for key, value in data.items():
                if key.startswith("validation_all_roc_auc_label_") and isinstance(value, (int, float)) and value > 0:
                    lbl_suffix = key.removeprefix("validation_all_roc_auc_")
                    n_pos = data.get(f"validation_all_n_pos_{lbl_suffix}", None)
                    scores.append(max(1e-6, float(value)))
                    weights.append(float(np.log1p(max(1, int(n_pos)))) if n_pos is not None else 1.0)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")

    if not scores:
        print("   [Warning] No valid validation_all_roc_auc_label_* scores found.")
        return 0.0
    w_arr = np.asarray(weights, dtype=np.float64)
    if np.all(w_arr > 0):
        return float(np.exp(np.average(np.log(np.asarray(scores, dtype=np.float64)), weights=w_arr)))
    return float(gmean(scores))


def cleanup_checkpoints(run_path: Path | None):
    if run_path and run_path.exists():
        for checkpoint_dir in run_path.rglob("checkpoint-*"):
            if checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _sync_lightweight_artifacts(src_run_path: Path, dst_run_path: Path, copy_checkpoints: bool):
    """
    Sync lightweight JSON/CSV metadata from node-local scratch to persistent OPTUNA_DIR,
    and copy heavy `checkpoint-*` folders ONLY when `copy_checkpoints=True` (new global best!).
    """
    if not src_run_path.exists() or src_run_path.resolve() == dst_run_path.resolve():
        return
    for item in src_run_path.rglob("*"):
        rel = item.relative_to(src_run_path)
        if item.is_symlink():
            target_dst = dst_run_path / rel
            target_dst.parent.mkdir(parents=True, exist_ok=True)
            if not target_dst.exists() and not target_dst.is_symlink():
                os.symlink(os.readlink(item), target_dst, target_is_directory=True)
            continue
        in_ckpt = any(part.startswith("checkpoint-") for part in rel.parts)
        if in_ckpt and not copy_checkpoints:
            continue
        target_dst = dst_run_path / rel
        if item.is_dir():
            target_dst.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_dst)


def _sync_symlink_or_dir(src: Path, dst: Path):
    """Point `dst` symlink at `src` atomically (only replacing existing symlinks, never real directories)."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(src.resolve(), dst, target_is_directory=True)


def _record_trial_failure_and_raise(exc: RuntimeError):
    with BEST_MODEL_LOCK:
        FAILURE_STATE["consecutive"] += 1
        consec = FAILURE_STATE["consecutive"]
    if consec >= cli_args.max_consecutive_failures:
        raise FatalWorkerError(
            f"Aborting worker after {consec} consecutive trial crashes. Last error: {exc}"
        ) from exc
    raise exc


def _reset_consecutive_failures():
    with BEST_MODEL_LOCK:
        FAILURE_STATE["consecutive"] = 0


def objective_pretrain(trial: optuna.Trial) -> float:
    """Stage 1: Optimize self-supervised pre-training hyperparameters with matched sample budgets."""
    gpu_id = GPU_QUEUE.get()
    persistent_trial_dir = OPTUNA_DIR / f"trial_{trial.number:03d}"
    work_trial_dir = (SCRATCH_TRIAL_ROOT / f"trial_{trial.number:03d}") if SCRATCH_TRIAL_ROOT else persistent_trial_dir
    try:
        trial_env = os.environ.copy()
        trial_env["CUDA_VISIBLE_DEVICES"] = gpu_id

        # 1. Masking probabilities (cap entity masking at 0.10 for causal next-event decoder)
        max_ent = 0.10 if IS_CAUSAL else 0.20
        p_entity = trial.suggest_float("mask_ent", 0.00, max_ent, step=0.05)
        p_attr = trial.suggest_float("mask_att", 0.05, 0.35, step=0.05)
        p_value = trial.suggest_float("mask_val", 0.15, 0.60, step=0.05)
        masking_rules = {"entity_id": p_entity, "attribute_id": p_attr, "value_id": p_value}

        # 2. Optimization & schedule hyperparameters (normalize step cadence by effective batch size!)
        lr_pre = trial.suggest_float("lr_pre", 5e-5, 8e-4, log=True)
        weight_decay_pre = trial.suggest_float("weight_decay_pre", 1e-4, 1e-1, log=True)
        lr_scheduler_pre = trial.suggest_categorical("lr_scheduler_pre", ["cosine", "constant_with_warmup"])
        warmup_base_pre = trial.suggest_int("warmup_base_pre", 50, 250, step=50)
        grad_accum_pre = trial.suggest_categorical("grad_accum_pre", [2, 4, 8])

        base_ga = int(CLI_CFG["pretrainer"].get("gradient_accumulation_steps", 4))
        base_eval_steps = int(CLI_CFG["pretrainer"].get("eval_steps", 100))
        base_max_steps = int(CLI_CFG["pretrainer"].get("max_steps", 5000))
        step_scale = base_ga / float(grad_accum_pre)
        scaled_eval_steps = max(25, int(round(base_eval_steps * step_scale))) if not DEBUG else 20
        scaled_warmup_steps = max(10, int(round(warmup_base_pre * step_scale))) if not DEBUG else 5
        scaled_max_steps = max(50, int(round(base_max_steps * step_scale))) if not DEBUG else 40

        # 3. Backbone capacity, receptive field & regularization
        dropout_emb = trial.suggest_float("dropout_embedding", 0.0, 0.25, step=0.05)
        dropout_att = trial.suggest_float("dropout_attention", 0.0, 0.25, step=0.05)
        cfg_model_args: dict = {
            "tie_word_embeddings": False,
            "dropout_embedding": dropout_emb,
            "dropout_attention": dropout_att,
        }
        if not IS_CAUSAL:
            loc_attn = trial.suggest_categorical("local_attention", [64, 128, 256])
            dropout_mlp = trial.suggest_float("dropout_mlp", 0.0, 0.25, step=0.05)
            cfg_model_args.update({
                "local_attention": loc_attn,
                "dropout_mlp": dropout_mlp,
            })

        portable_overrides = {
            "run_id": cli_args.export_run_id,
            "data_collator": {
                "mlm_masking_rules": masking_rules,
                "use_numerical_embedding": (VAL_TAG == "raw"),
            },
            "pretrainer": {
                "learning_rate": lr_pre,
                "weight_decay": weight_decay_pre,
                "lr_scheduler_type": lr_scheduler_pre,
                "warmup_steps": scaled_warmup_steps,
                "gradient_accumulation_steps": grad_accum_pre,
                "eval_steps": scaled_eval_steps,
                "save_steps": scaled_eval_steps,
                "max_steps": scaled_max_steps,
            },
            "model": {
                "config_args": cfg_model_args,
                "embedding_layer_config": {"use_numerical_embedding": (VAL_TAG == "raw")},
            },
        }
        work_overrides = {
            **portable_overrides,
            "result_dir": str(work_trial_dir),
            "data_split_type": SPLIT_TYPE,
            "do_clustering": False,
            "pretrainer": {
                **portable_overrides["pretrainer"],
                "seed": 1234 + trial.number,
            },
        }
        work_overrides.pop("run_id", None)
        persistent_overrides = {**work_overrides, "result_dir": str(persistent_trial_dir)}

        arch_str = f"W={cfg_model_args['local_attention']}, " if not IS_CAUSAL else ""
        print(
            f"\n[Pretrain Trial {trial.number}] GPU {gpu_id}: {arch_str}"
            f"mask=({p_entity:.2f},{p_attr:.2f},{p_value:.2f}), lr={lr_pre:.2e} ({lr_scheduler_pre}), "
            f"wd={weight_decay_pre:.2e}, ga={grad_accum_pre} (eval={scaled_eval_steps}, max={scaled_max_steps}), "
            f"drop=({dropout_emb:.2f},{dropout_att:.2f}"
            + (f",{cfg_model_args['dropout_mlp']:.2f})" if not IS_CAUSAL else ")")
        )
        try:
            run_subprocess(
                "scripts/train_mlm.py",
                work_overrides,
                env=trial_env,
                error_log_path=persistent_trial_dir / "trial_error.log",
            )
        except RuntimeError as e:
            print(f"[Pretrain Trial {trial.number}] Failed: {e}")
            _record_trial_failure_and_raise(e)

        _reset_consecutive_failures()
        run_id = get_run_id(masking_rules)
        work_run_path = work_trial_dir / SPLIT_TYPE / run_id
        persistent_run_path = persistent_trial_dir / SPLIT_TYPE / run_id
        score = evaluate_pretrain_trial(work_run_path / "pretraining")
        print(f"[Pretrain Trial {trial.number}] GPU {gpu_id} -> Standardized Ref MLM Score = {score:.4f}")

        is_new_best = False
        path_to_delete = None
        with BEST_MODEL_LOCK:
            study_best = -float("inf")
            try:
                study_best = float(trial.study.best_value)
            except Exception:
                pass
            threshold_best = max(BEST_MODEL_INFO["score"], study_best)
            if score > threshold_best:
                is_new_best = True
                BEST_MODEL_INFO["score"] = score
                BEST_MODEL_INFO["path"] = persistent_run_path
                BEST_MODEL_INFO["params"] = trial.params
                BEST_MODEL_INFO["overrides"] = persistent_overrides
                BEST_MODEL_INFO["portable_overrides"] = portable_overrides
            elif SCRATCH_TRIAL_ROOT is None:
                path_to_delete = persistent_run_path

            _sync_lightweight_artifacts(work_run_path, persistent_run_path, copy_checkpoints=is_new_best)
            if path_to_delete:
                cleanup_checkpoints(path_to_delete)

        trial.set_user_attr("run_path", str(persistent_run_path))
        trial.set_user_attr("portable_overrides", portable_overrides)
        return score
    finally:
        if SCRATCH_TRIAL_ROOT is not None and work_trial_dir.exists():
            shutil.rmtree(work_trial_dir, ignore_errors=True)
        GPU_QUEUE.put(gpu_id)


def objective_finetune(trial: optuna.Trial) -> float:
    """Stage 2: Optimize multi-task fine-tuning hyperparameters from the fixed Stage-1 pretrained checkpoint."""
    gpu_id = GPU_QUEUE.get()
    persistent_trial_dir = OPTUNA_DIR / f"trial_{trial.number:03d}"
    work_trial_dir = (SCRATCH_TRIAL_ROOT / f"trial_{trial.number:03d}") if SCRATCH_TRIAL_ROOT else persistent_trial_dir
    try:
        trial_env = os.environ.copy()
        trial_env["CUDA_VISIBLE_DEVICES"] = gpu_id

        source_pt_dir, _ = resolve_source_pretrained_dir()
        default_rule = CLI_CFG["data_collator"]["mlm_masking_rules"]
        run_id = get_run_id(default_rule)

        # Link the shared pretrained checkpoint into both scratch & persistent trial directories
        for base_td in ({work_trial_dir, persistent_trial_dir}):
            pt_link = base_td / SPLIT_TYPE / run_id / "pretraining"
            pt_link.parent.mkdir(parents=True, exist_ok=True)
            if not pt_link.exists() and not pt_link.is_symlink():
                os.symlink(source_pt_dir.resolve(), pt_link, target_is_directory=True)

        # 1. Optimization hyperparameters (normalize step cadence by effective batch size!)
        lr_fine = trial.suggest_float("lr_fine", 1e-5, 2e-4, log=True)
        weight_decay_fine = trial.suggest_float("weight_decay_fine", 1e-4, 1e-1, log=True)
        warmup_base_fine = trial.suggest_int("warmup_base_fine", 20, 140, step=20)
        grad_accum_fine = trial.suggest_categorical("grad_accum_fine", [2, 4, 8])

        base_ga = int(CLI_CFG["finetuner"].get("gradient_accumulation_steps", 4))
        base_eval_steps = int(CLI_CFG["finetuner"].get("eval_steps", 50))
        base_max_steps = int(CLI_CFG["finetuner"].get("max_steps", 10000))
        step_scale = base_ga / float(grad_accum_fine)
        scaled_eval_steps = max(20, int(round(base_eval_steps * step_scale))) if not DEBUG else 15
        scaled_warmup_steps = max(10, int(round(warmup_base_fine * step_scale))) if not DEBUG else 5
        scaled_max_steps = max(40, int(round(base_max_steps * step_scale))) if not DEBUG else 30

        # 2. Regularization hyperparameters (architecture-aware: no dead no-op params)
        dropout_emb = trial.suggest_float("dropout_embedding", 0.0, 0.25, step=0.05)
        dropout_att = trial.suggest_float("dropout_attention", 0.0, 0.25, step=0.05)
        dropout_cls = trial.suggest_float("dropout_classifier", 0.0, 0.40, step=0.05)
        cfg_dropouts = {
            "dropout_embedding": dropout_emb,
            "dropout_attention": dropout_att,
            "dropout_classifier": dropout_cls,
        }
        if not IS_CAUSAL:
            cfg_dropouts["dropout_mlp"] = trial.suggest_float("dropout_mlp", 0.0, 0.25, step=0.05)

        # 3. Unweighted BCE ("ce") for calibrated probabilities + hierarchical random-effects head aggregation
        loss_name = "ce"
        es_mean = CLI_CFG.get("early_stopping_mean", "random_effects")
        portable_overrides = {
            "run_id": cli_args.export_run_id,
            "early_stopping_mean": es_mean,
            "data_collator": {"use_numerical_embedding": (VAL_TAG == "raw")},
            "finetuner": {
                "learning_rate": lr_fine,
                "weight_decay": weight_decay_fine,
                "warmup_steps": scaled_warmup_steps,
                "gradient_accumulation_steps": grad_accum_fine,
                "eval_steps": scaled_eval_steps,
                "save_steps": scaled_eval_steps,
                "max_steps": scaled_max_steps,
                "loss_name": loss_name,
            },
            "model": {
                "config_args": cfg_dropouts,
                "embedding_layer_config": {"use_numerical_embedding": (VAL_TAG == "raw")},
            },
        }
        work_overrides = {
            **portable_overrides,
            "result_dir": str(work_trial_dir),
            "data_split_type": SPLIT_TYPE,
            "n_bootstrap_eval": 20,
            "finetuner": {
                **portable_overrides["finetuner"],
                "seed": 1234 + trial.number,
            },
        }
        work_overrides.pop("run_id", None)
        persistent_overrides = {**work_overrides, "result_dir": str(persistent_trial_dir)}

        print(
            f"\n[Finetune Trial {trial.number}] GPU {gpu_id} ({TASKS_ARG}): "
            f"lr={lr_fine:.2e}, wd={weight_decay_fine:.2e}, ga={grad_accum_fine} (eval_steps={scaled_eval_steps}), "
            f"drop={cfg_dropouts}"
        )
        try:
            run_subprocess(
                "scripts/train_classification.py",
                work_overrides,
                env=trial_env,
                extra_args=["--tasks", TASKS_ARG, "--skip-test"],
                error_log_path=persistent_trial_dir / "trial_error.log",
            )
        except RuntimeError as e:
            print(f"[Finetune Trial {trial.number}] Failed: {e}")
            _record_trial_failure_and_raise(e)

        _reset_consecutive_failures()
        work_run_path = work_trial_dir / SPLIT_TYPE / run_id
        persistent_run_path = persistent_trial_dir / SPLIT_TYPE / run_id
        score = evaluate_finetune_trial(work_run_path)
        print(f"[Finetune Trial {trial.number}] GPU {gpu_id} -> Validation Log-Support GMean ROC-AUC = {score:.4f}")

        is_new_best = False
        path_to_delete = None
        with BEST_MODEL_LOCK:
            study_best = -float("inf")
            try:
                study_best = float(trial.study.best_value)
            except Exception:
                pass
            threshold_best = max(BEST_MODEL_INFO["score"], study_best)
            if score > threshold_best:
                is_new_best = True
                BEST_MODEL_INFO["score"] = score
                BEST_MODEL_INFO["path"] = persistent_run_path
                BEST_MODEL_INFO["params"] = trial.params
                BEST_MODEL_INFO["overrides"] = persistent_overrides
                BEST_MODEL_INFO["portable_overrides"] = portable_overrides
            elif SCRATCH_TRIAL_ROOT is None:
                path_to_delete = persistent_run_path

            _sync_lightweight_artifacts(work_run_path, persistent_run_path, copy_checkpoints=is_new_best)
            if path_to_delete:
                cleanup_checkpoints(path_to_delete)

        trial.set_user_attr("run_path", str(persistent_run_path))
        trial.set_user_attr("overrides", persistent_overrides)
        trial.set_user_attr("portable_overrides", portable_overrides)
        return score
    finally:
        if SCRATCH_TRIAL_ROOT is not None and work_trial_dir.exists():
            shutil.rmtree(work_trial_dir, ignore_errors=True)
        GPU_QUEUE.put(gpu_id)


def main():
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    log_file = str(OPTUNA_DIR / f"{STUDY_NAME}_journal.log")

    if cli_args.compact_journal or cli_args.n_trials == 0:
        compact_journal_duplicate_studies(log_file)

    slurm_jid = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
    sampler = TPESampler(
        seed=(42 + slurm_jid * 31 + os.getpid()) % (2**31 - 1),
        n_startup_trials=12,
        multivariate=True,
        constant_liar=True,
    )
    study, storage = load_or_create_study_once(log_file=log_file, study_name=STUDY_NAME, sampler=sampler)

    if cli_args.reap_running:
        reap_stale_running_trials(study, storage)

    if cli_args.init_only:
        print(f"[Init] Study '{STUDY_NAME}' ready in {log_file} ({len(study.trials)} existing trials).")
        return

    if len(study.trials) > 0:
        try:
            bt = study.best_trial
            BEST_MODEL_INFO["score"] = float(bt.value)
            if bt.user_attrs.get("run_path"):
                BEST_MODEL_INFO["path"] = Path(bt.user_attrs["run_path"])
            BEST_MODEL_INFO["params"] = bt.params
            BEST_MODEL_INFO["overrides"] = bt.user_attrs.get("overrides")
            BEST_MODEL_INFO["portable_overrides"] = bt.user_attrs.get("portable_overrides")
        except ValueError:
            pass

    n_complete_before = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    print("=" * 70)
    print(f"Starting Optuna Stage='{STAGE}' | Model='{BASE_RESULTS_DIR.name}' ({'causal' if IS_CAUSAL else 'encoder'})")
    print(
        f"Split='{SPLIT_TYPE}' | ValTag='{VAL_TAG}' | WorkerCap={N_TRIALS} | "
        f"GlobalTarget={TARGET_TRIALS} (currently {n_complete_before} COMPLETE) | GPUs={NUM_GPUS} ({GPU_IDS})"
    )
    print(f"Scratch Trial Dir: {SCRATCH_TRIAL_ROOT if SCRATCH_TRIAL_ROOT else 'Disabled (using OPTUNA_DIR)'}")
    print(f"Journal Storage: {log_file}")
    print("=" * 70)

    obj_fn = objective_pretrain if STAGE == "pretrain" else objective_finetune
    should_run = (N_TRIALS is None or N_TRIALS > 0) and (TARGET_TRIALS <= 0 or n_complete_before < TARGET_TRIALS)
    if should_run:
        callbacks = []
        if TARGET_TRIALS > 0:
            callbacks.append(MaxTrialsCallback(TARGET_TRIALS, states=(TrialState.COMPLETE,)))
        try:
            study.optimize(
                obj_fn,
                n_trials=N_TRIALS,
                n_jobs=NUM_GPUS,
                callbacks=callbacks,
                catch=(RuntimeError,),
            )
        finally:
            if SCRATCH_TRIAL_ROOT is not None and SCRATCH_TRIAL_ROOT.exists():
                shutil.rmtree(SCRATCH_TRIAL_ROOT, ignore_errors=True)
    elif N_TRIALS != 0 and TARGET_TRIALS > 0 and n_complete_before >= TARGET_TRIALS:
        print(f"[Skip] Study already reached {n_complete_before}/{TARGET_TRIALS} COMPLETE trials.")

    # Refresh global winner from shared JournalStorage (handles resumed or multi-job workers cleanly)
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not completed_trials:
        raise RuntimeError(f"No COMPLETE trials found in study '{STUDY_NAME}' ({log_file}).")

    best_trial = study.best_trial
    best_run_path = Path(best_trial.user_attrs["run_path"]) if best_trial.user_attrs.get("run_path") else BEST_MODEL_INFO["path"]
    best_overrides = best_trial.user_attrs.get("overrides") or BEST_MODEL_INFO["overrides"]
    best_portable = best_trial.user_attrs.get("portable_overrides") or BEST_MODEL_INFO["portable_overrides"]

    print("\n" + "=" * 70)
    print(
        f"Optimization summary for stage='{STAGE}' "
        f"({len(completed_trials)} COMPLETE trials | Global Best Trial #{best_trial.number})"
    )
    print("=" * 70)
    print("Best parameters:")
    print(json.dumps(best_trial.params, indent=4))
    print(f"Best validation score: {study.best_value:.4f}")

    if cli_args.skip_final_eval:
        print("[Worker] --skip-final-eval set; deferring canonical linking & post-hoc evaluation to finalizer.")
        return

    # Prune any leftover runner-up trial checkpoints left by parallel workers
    if best_run_path is not None:
        best_trial_dir = OPTUNA_DIR / f"trial_{best_trial.number:03d}"
        for td in sorted(OPTUNA_DIR.glob("trial_*")):
            if td.is_dir() and td.resolve() != best_trial_dir.resolve():
                cleanup_checkpoints(td)

    canonical_base = BASE_RESULTS_DIR / SPLIT_TYPE / EXPORT_RUN_ID

    if STAGE == "pretrain" and best_run_path is not None and (best_run_path / "pretraining").exists():
        _sync_symlink_or_dir(best_run_path / "pretraining", canonical_base / "pretraining")
        print(f"Linked winning pretraining directory -> {canonical_base / 'pretraining'}")

    # For Stage 2 (finetune), run full-FUP validation/test scoring + plots ONCE on the winning checkpoint
    if STAGE == "finetune" and best_run_path is not None and best_overrides is not None:
        ft_root = best_run_path / "finetuning"
        has_ckpts = bool(list(ft_root.rglob("checkpoint-*"))) if ft_root.exists() else False
        req_tasks = [t.strip() for t in TASKS_ARG.split(",") if t.strip()]
        already_scored = bool(req_tasks) and all(
            bool(list((ft_root / t).rglob("test_predictions.csv")))
            for t in req_tasks
            if (ft_root / t).exists()
        ) and all((ft_root / t).exists() for t in req_tasks)
        if has_ckpts and not already_scored:
            print(f"\n>>> Running final full-FUP evaluation (--eval-only) on winning checkpoint: {best_run_path}")
            final_env = os.environ.copy()
            final_env["CUDA_VISIBLE_DEVICES"] = GPU_IDS[0]
            try:
                run_subprocess(
                    "scripts/train_classification.py",
                    best_overrides,
                    env=final_env,
                    extra_args=["--tasks", TASKS_ARG, "--eval-only"],
                    error_log_path=OPTUNA_DIR / "final_eval_error.log",
                )
            except RuntimeError as e:
                print(f"[Warning] Post-Optuna --eval-only pass failed: {e}")
        # Link canonical pretraining + per-task finetuning directories for downstream analysis scripts
        source_pt_dir, _ = resolve_source_pretrained_dir()
        _sync_symlink_or_dir(source_pt_dir, canonical_base / "pretraining")
        if (best_run_path / "finetuning").exists():
            for task_dir in sorted((best_run_path / "finetuning").iterdir()):
                if task_dir.is_dir():
                    _sync_symlink_or_dir(task_dir, canonical_base / "finetuning" / task_dir.name)
            print(f"Linked winning finetuning task(s) -> {canonical_base / 'finetuning'}")

    summary = {
        "stage": STAGE,
        "model_family": BASE_RESULTS_DIR.name,
        "is_causal": IS_CAUSAL,
        "split": SPLIT_TYPE,
        "val_tag": VAL_TAG,
        "export_run_id": EXPORT_RUN_ID,
        "tasks": TASKS_ARG if STAGE == "finetune" else "mlm",
        "n_complete_trials": len(completed_trials),
        "best_value": float(study.best_value),
        "best_trial_number": int(best_trial.number),
        "best_params": best_trial.params,
        "portable_overrides": best_portable,
        "best_run_path": str(best_run_path) if best_run_path else None,
        "best_pretraining_dir": (
            str(best_run_path / "pretraining")
            if (STAGE == "pretrain" and best_run_path)
            else None
        ),
    }
    best_config_path = OPTUNA_DIR / f"best_{STAGE}_config.json"
    with open(best_config_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved best configuration summary to: {best_config_path}")


if __name__ == "__main__":
    main()
