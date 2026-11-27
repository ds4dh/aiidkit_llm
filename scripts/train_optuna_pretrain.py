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
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
try:
    from optuna.storages.journal import JournalFileBackend as _JournalFileBackend
except ImportError:
    from optuna.storages import JournalFileStorage as _JournalFileBackend
from src.utils import apply_config_overrides
from scripts.script_utils import format_pretrain_run_id, get_value_embedding_tag


class FatalWorkerError(BaseException):
    """Raised when a worker hits repeated consecutive trial crashes and should abort fast."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for MLM pre-training of the t-EAV Transformer backbone."
    )
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training_compact.yaml")
    parser.add_argument("--split", type=str, default=None, help="Override data_split_type (default: from config).")
    parser.add_argument(
        "--export-run-id",
        type=str,
        default="optuna_pt_c8x384",
        help="Canonical run_id under results/<family>/<split>/<run_id>_<val_tag>/pretraining where the winning checkpoint is linked.",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=40,
        help="Global target of COMPLETE trials across all parallel workers (default: 40; 0 disables global cap).",
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
        help="Abort worker if this many consecutive trials crash.",
    )
    parser.add_argument(
        "--reap-running",
        action="store_true",
        help="Mark any orphaned RUNNING trials (from dead workers) as FAIL and clean partial checkpoints.",
    )
    parser.add_argument(
        "--compact-journal",
        action="store_true",
        help="Deduplicate redundant CREATE_STUDY (op_code=0) entries in the journal file.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize the Optuna journal study once (if needed) and exit immediately.",
    )
    parser.add_argument(
        "--skip-final-export",
        action="store_true",
        help="Skip post-study checkpoint linking & summary export (use on parallel Slurm workers when a finalizer runs afterany).",
    )
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Optional JSON or key=value config overrides.")
    parser.add_argument("--debug", "-d", action="store_true", help="Run a quick debug check with capped steps and trials.")
    return parser.parse_args()


cli_args = parse_args()
DEBUG = cli_args.debug
CONFIG_PATH = cli_args.config
with open(CONFIG_PATH, "r") as f:
    CLI_CFG = yaml.safe_load(f)
if cli_args.overrides:
    CLI_CFG = apply_config_overrides(CLI_CFG, cli_args.overrides)

SPLIT_TYPE = cli_args.split or CLI_CFG.get("data_split_type", "random_split")
VAL_TAG = get_value_embedding_tag(CLI_CFG)
EXPORT_RUN_ID = (
    f"{cli_args.export_run_id}_{VAL_TAG}"
    if not cli_args.export_run_id.endswith(f"_{VAL_TAG}")
    else cli_args.export_run_id
)
BASE_RESULTS_DIR = Path(CLI_CFG["result_dir"])

STUDY_PREFIX = f"{SPLIT_TYPE}_{cli_args.export_run_id}_{VAL_TAG}"
OPTUNA_DIR = BASE_RESULTS_DIR / "optuna_pretrain" / STUDY_PREFIX
TARGET_TRIALS = 2 if DEBUG else int(cli_args.target_trials)
N_TRIALS = cli_args.n_trials
STUDY_NAME = f"pretraining_hp_{STUDY_PREFIX}"

BEST_MODEL_LOCK = threading.Lock()
BEST_MODEL_INFO = {
    "score": float("inf"),  # minimizing standardized ref_eval_loss
    "path": None,
    "params": None,
    "overrides": None,
    "portable_overrides": None,
    "ref_metrics": None,
}
FAILURE_STATE = {"consecutive": 0}

if cli_args.init_only or cli_args.n_trials == 0:
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
    """Prefer node-local fast scratch for transient MLM trial checkpoints."""
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
                    root = cand / f"optuna_{BASE_RESULTS_DIR.name}_pretrain_{SPLIT_TYPE}_{os.getpid()}"
                    root.mkdir(parents=True, exist_ok=True)
                    return root
        except OSError:
            continue
    return None


SCRATCH_TRIAL_ROOT = _resolve_scratch_trial_root()


def compact_journal_duplicate_studies(log_file: str) -> int:
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
        storage = JournalStorage(_JournalFileBackend(log_file))
        try:
            study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
            return study, storage
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=False,
                sampler=sampler,
            )
            # Enqueue the active YAML pre-training defaults as Trial #0 baseline anchor
            pt_defaults = CLI_CFG.get("pretrainer", {}) or {}
            m_defaults = (CLI_CFG.get("model", {}) or {}).get("config_args", {}) or {}
            mask_defaults = (CLI_CFG.get("data_collator", {}) or {}).get("mlm_masking_rules", {}) or {}

            def _clamp(val, lo, hi):
                return max(lo, min(hi, val))

            def _snap(val, lo, hi, step):
                return round(_clamp(round(val / step) * step, lo, hi), 6)

            base_anchor = {
                "mask_entity": float(mask_defaults.get("entity_id", 0.0)) if float(mask_defaults.get("entity_id", 0.0)) in (0.0, 0.05, 0.10, 0.15) else 0.0,
                "mask_attribute": float(_snap(float(mask_defaults.get("attribute_id", 0.15)), 0.05, 0.35, 0.05)),
                "mask_value": float(_snap(float(mask_defaults.get("value_id", 0.60)), 0.30, 0.80, 0.05)),
                "lr_pre": float(_clamp(float(pt_defaults.get("learning_rate", 3.0e-4)), 1.0e-4, 1.5e-3)),
                "weight_decay_pre": float(_clamp(float(pt_defaults.get("weight_decay", 2.0e-2)), 1.0e-3, 1.0e-1)),
                "grad_accum_pre": int(pt_defaults.get("gradient_accumulation_steps", 4)) if int(pt_defaults.get("gradient_accumulation_steps", 4)) in (2, 4, 8) else 4,
                "warmup_base_pre": int(_snap(float(pt_defaults.get("warmup_steps", 100)), 40, 240, 20)),
                "lr_scheduler_type": str(pt_defaults.get("lr_scheduler_type", "cosine")) if str(pt_defaults.get("lr_scheduler_type", "cosine")) in ("cosine", "linear") else "cosine",
                "dropout_embedding": float(_snap(float(m_defaults.get("dropout_embedding", 0.0)), 0.0, 0.15, 0.05)),
                "dropout_attention": float(_snap(float(m_defaults.get("dropout_attention", 0.15)), 0.0, 0.25, 0.05)),
                "dropout_mlp": float(_snap(float(m_defaults.get("dropout_mlp", 0.15)), 0.0, 0.25, 0.05)),
            }
            study.enqueue_trial(base_anchor)
            return study, storage
    finally:
        if acquired:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass


def cleanup_checkpoints(run_path: Path | None):
    if run_path and run_path.exists():
        for checkpoint_dir in run_path.rglob("checkpoint-*"):
            if checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)


def reap_stale_running_trials(study: optuna.Study, storage: JournalStorage) -> int:
    reaped = 0
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        cleanup_checkpoints(OPTUNA_DIR / f"trial_{t.number:03d}")
        reaped += 1
    if reaped > 0:
        print(f"[Reaper] Marked {reaped} orphaned RUNNING trial(s) as FAIL and cleaned partial checkpoints.")
    return reaped


def _sync_lightweight_artifacts(src_run_path: Path, dst_run_path: Path, copy_checkpoints: bool):
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


def run_subprocess(script_path: str, overrides: dict, env: dict, error_log_path: Path | None = None):
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        CONFIG_PATH,
    ]
    for ov in (cli_args.overrides or []):
        cmd.extend(["--overrides", ov])
    cmd.extend(["--overrides", json.dumps(overrides), "--silent"])

    child_env = env.copy()
    if not DEBUG:
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


def evaluate_pretrain_trial(run_path: Path) -> tuple[float, dict]:
    """Read standardized reference-mask validation metrics from `pretraining_results.json`."""
    res_path = run_path / "pretraining" / "pretraining_results.json"
    if not res_path.exists():
        raise RuntimeError(f"Missing pretraining_results.json at {res_path}")
    with open(res_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    ref_loss = metrics.get("ref_eval_loss", None)
    if ref_loss is None or not isinstance(ref_loss, (int, float)) or ref_loss <= 0 or ref_loss != ref_loss:
        raise RuntimeError(f"Invalid ref_eval_loss={ref_loss} in {res_path}")
    return float(ref_loss), metrics


def objective_pretrain(trial: optuna.Trial) -> float:
    """Optimize MLM pre-training hyperparameters evaluated on the fixed e00-a15-v60 reference validation mask."""
    gpu_id = GPU_QUEUE.get()
    persistent_trial_dir = OPTUNA_DIR / f"trial_{trial.number:03d}"
    work_trial_dir = (SCRATCH_TRIAL_ROOT / f"trial_{trial.number:03d}") if SCRATCH_TRIAL_ROOT else persistent_trial_dir
    try:
        trial_env = os.environ.copy()
        trial_env["CUDA_VISIBLE_DEVICES"] = gpu_id

        # 1. Masking probabilities (training mask; all trials are evaluated on fixed e00-a15-v60 ref mask)
        mask_entity = trial.suggest_categorical("mask_entity", [0.0, 0.05, 0.10, 0.15])
        mask_attribute = trial.suggest_float("mask_attribute", 0.05, 0.35, step=0.05)
        mask_value = trial.suggest_float("mask_value", 0.30, 0.80, step=0.05)
        masking_rules = {
            "entity_id": float(mask_entity),
            "attribute_id": float(round(mask_attribute, 4)),
            "value_id": float(round(mask_value, 4)),
        }

        # 2. Optimization hyperparameters (normalize step cadence by effective batch size)
        lr_pre = trial.suggest_float("lr_pre", 1.0e-4, 1.5e-3, log=True)
        weight_decay_pre = trial.suggest_float("weight_decay_pre", 1.0e-3, 1.0e-1, log=True)
        grad_accum_pre = trial.suggest_categorical("grad_accum_pre", [2, 4, 8])
        warmup_base_pre = trial.suggest_int("warmup_base_pre", 40, 240, step=20)
        lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["cosine", "linear"])

        base_ga = int(CLI_CFG["pretrainer"].get("gradient_accumulation_steps", 4))
        base_eval_steps = int(CLI_CFG["pretrainer"].get("eval_steps", 100))
        base_max_steps = int(CLI_CFG["pretrainer"].get("max_steps", 6000))
        step_scale = base_ga / float(grad_accum_pre)
        scaled_eval_steps = max(25, int(round(base_eval_steps * step_scale))) if not DEBUG else 10
        scaled_warmup_steps = max(20, int(round(warmup_base_pre * step_scale))) if not DEBUG else 5
        scaled_max_steps = max(100, int(round(base_max_steps * step_scale))) if not DEBUG else 20

        # 3. Regularization hyperparameters
        dropout_emb = trial.suggest_float("dropout_embedding", 0.0, 0.15, step=0.05)
        dropout_att = trial.suggest_float("dropout_attention", 0.0, 0.25, step=0.05)
        dropout_mlp = trial.suggest_float("dropout_mlp", 0.0, 0.25, step=0.05)
        cfg_dropouts = {
            "dropout_embedding": dropout_emb,
            "dropout_attention": dropout_att,
            "dropout_mlp": dropout_mlp,
        }

        use_num_emb = VAL_TAG.startswith("raw")
        portable_overrides = {
            "run_id": cli_args.export_run_id,
            "data_collator": {
                "use_numerical_embedding": use_num_emb,
                "mlm_masking_rules": masking_rules,
            },
            "pretrainer": {
                "learning_rate": lr_pre,
                "lr_scheduler_type": lr_scheduler_type,
                "weight_decay": weight_decay_pre,
                "warmup_steps": scaled_warmup_steps,
                "gradient_accumulation_steps": grad_accum_pre,
                "eval_steps": scaled_eval_steps,
                "save_steps": scaled_eval_steps,
                "max_steps": scaled_max_steps,
            },
            "model": {
                "config_args": cfg_dropouts,
                "embedding_layer_config": {"use_numerical_embedding": use_num_emb},
            },
        }
        # Keep deterministic run_id inside isolated trial folder so train_mlm.py output path is unambiguous
        trial_run_id = format_pretrain_run_id(CLI_CFG)
        work_overrides = {
            **portable_overrides,
            "run_id": CLI_CFG.get("run_id") or trial_run_id,
            "result_dir": str(work_trial_dir),
            "data_split_type": SPLIT_TYPE,
            "pretrainer": {
                **portable_overrides["pretrainer"],
                "seed": 1234 + trial.number,
            },
        }
        persistent_overrides = {**work_overrides, "result_dir": str(persistent_trial_dir)}

        print(
            f"\n[Pretrain Trial {trial.number}] GPU {gpu_id}: "
            f"mask=e{int(round(mask_entity*100)):02d}-a{int(round(mask_attribute*100)):02d}-v{int(round(mask_value*100)):02d}, "
            f"lr={lr_pre:.2e} ({lr_scheduler_type}), wd={weight_decay_pre:.2e}, warmup={scaled_warmup_steps}, "
            f"ga={grad_accum_pre} (eval_steps={scaled_eval_steps}, max_steps={scaled_max_steps}), drop={cfg_dropouts}"
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
        work_run_path = work_trial_dir / SPLIT_TYPE / trial_run_id
        persistent_run_path = persistent_trial_dir / SPLIT_TYPE / trial_run_id
        ref_loss, ref_metrics = evaluate_pretrain_trial(work_run_path)
        ref_acc = float(ref_metrics.get("ref_eval_mlm_accuracy", 0.0))
        ref_disc = float(ref_metrics.get("ref_eval_mlm_discrete_accuracy", 0.0))
        ref_cont = float(ref_metrics.get("ref_eval_mlm_continuous_accuracy", 0.0))
        runtime_s = float(ref_metrics.get("train_runtime", 0.0))
        print(
            f"[Pretrain Trial {trial.number}] GPU {gpu_id} -> "
            f"ref_eval_loss={ref_loss:.4f} | ref_mlm_acc={ref_acc:.4f} "
            f"(disc={ref_disc:.4f}, cont={ref_cont:.4f}) | {runtime_s/60:.1f} min"
        )

        is_new_best = False
        path_to_delete = None
        with BEST_MODEL_LOCK:
            study_best = float("inf")
            try:
                study_best = float(trial.study.best_value)
            except Exception:
                pass
            threshold_best = min(BEST_MODEL_INFO["score"], study_best)
            if ref_loss < threshold_best:
                is_new_best = True
                BEST_MODEL_INFO["score"] = ref_loss
                BEST_MODEL_INFO["path"] = persistent_run_path
                BEST_MODEL_INFO["params"] = trial.params
                BEST_MODEL_INFO["overrides"] = persistent_overrides
                BEST_MODEL_INFO["portable_overrides"] = portable_overrides
                BEST_MODEL_INFO["ref_metrics"] = ref_metrics
            elif SCRATCH_TRIAL_ROOT is None:
                path_to_delete = persistent_run_path

            _sync_lightweight_artifacts(work_run_path, persistent_run_path, copy_checkpoints=is_new_best)
            if path_to_delete:
                cleanup_checkpoints(path_to_delete)

        trial.set_user_attr("run_path", str(persistent_run_path))
        trial.set_user_attr("overrides", persistent_overrides)
        trial.set_user_attr("portable_overrides", portable_overrides)
        trial.set_user_attr("ref_eval_mlm_accuracy", ref_acc)
        trial.set_user_attr("ref_eval_mlm_discrete_accuracy", ref_disc)
        trial.set_user_attr("ref_eval_mlm_continuous_accuracy", ref_cont)
        trial.set_user_attr("train_runtime", runtime_s)
        return ref_loss
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
        n_startup_trials=10,
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
    print(f"Starting Pre-Training Optuna | Model='{BASE_RESULTS_DIR.name}' | Config='{CONFIG_PATH}'")
    print(
        f"Split='{SPLIT_TYPE}' | ValTag='{VAL_TAG}' | ExportRunID='{EXPORT_RUN_ID}' | "
        f"WorkerCap={N_TRIALS} | GlobalTarget={TARGET_TRIALS} (currently {n_complete_before} COMPLETE) | GPUs={NUM_GPUS} ({GPU_IDS})"
    )
    print(f"Scratch Trial Dir: {SCRATCH_TRIAL_ROOT if SCRATCH_TRIAL_ROOT else 'Disabled (using OPTUNA_DIR)'}")
    print(f"Journal Storage: {log_file}")
    print("=" * 70)

    should_run = (N_TRIALS is None or N_TRIALS > 0) and (TARGET_TRIALS <= 0 or n_complete_before < TARGET_TRIALS)
    if should_run:
        callbacks = []
        if TARGET_TRIALS > 0:
            callbacks.append(MaxTrialsCallback(TARGET_TRIALS, states=(TrialState.COMPLETE,)))
        try:
            study.optimize(
                objective_pretrain,
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

    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not completed_trials:
        raise RuntimeError(f"No COMPLETE trials found in study '{STUDY_NAME}' ({log_file}).")

    best_trial = study.best_trial
    best_run_path = Path(best_trial.user_attrs["run_path"]) if best_trial.user_attrs.get("run_path") else BEST_MODEL_INFO["path"]
    best_portable = best_trial.user_attrs.get("portable_overrides") or BEST_MODEL_INFO["portable_overrides"]

    print("\n" + "=" * 70)
    print(
        f"Pre-Training Optuna Summary "
        f"({len(completed_trials)} COMPLETE trials | Global Best Trial #{best_trial.number})"
    )
    print("=" * 70)
    print("Best parameters:")
    print(json.dumps(best_trial.params, indent=4))
    print(
        f"Best standardized ref_eval_loss: {study.best_value:.4f} | "
        f"ref_eval_mlm_accuracy: {best_trial.user_attrs.get('ref_eval_mlm_accuracy', 0.0):.4f}"
    )

    if cli_args.skip_final_export:
        print("[Worker] --skip-final-export set; deferring canonical linking & summary export to finalizer.")
        return

    # Prune runner-up checkpoints left by parallel workers
    if best_run_path is not None:
        best_trial_dir = OPTUNA_DIR / f"trial_{best_trial.number:03d}"
        for td in sorted(OPTUNA_DIR.glob("trial_*")):
            if td.is_dir() and td.resolve() != best_trial_dir.resolve():
                cleanup_checkpoints(td)

        canonical_pt = BASE_RESULTS_DIR / SPLIT_TYPE / EXPORT_RUN_ID / "pretraining"
        _sync_symlink_or_dir(best_run_path / "pretraining", canonical_pt)
        print(f"Linked winning pretrained checkpoint -> {canonical_pt}")

    # Also write a compact ranked CSV of all completed trials for quick inspection
    rows = []
    for t in sorted(completed_trials, key=lambda x: x.value):
        row = {
            "trial": t.number,
            "ref_eval_loss": float(t.value),
            "ref_eval_mlm_accuracy": float(t.user_attrs.get("ref_eval_mlm_accuracy", 0.0)),
            "ref_eval_mlm_discrete_accuracy": float(t.user_attrs.get("ref_eval_mlm_discrete_accuracy", 0.0)),
            "ref_eval_mlm_continuous_accuracy": float(t.user_attrs.get("ref_eval_mlm_continuous_accuracy", 0.0)),
            "train_runtime_s": float(t.user_attrs.get("train_runtime", 0.0)),
            **t.params,
        }
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OPTUNA_DIR / "pretrain_trials_ranked.csv", index=False)

    summary = {
        "stage": "pretrain",
        "model_family": BASE_RESULTS_DIR.name,
        "split": SPLIT_TYPE,
        "val_tag": VAL_TAG,
        "export_run_id": EXPORT_RUN_ID,
        "n_complete_trials": len(completed_trials),
        "best_ref_eval_loss": float(study.best_value),
        "best_ref_eval_mlm_accuracy": float(best_trial.user_attrs.get("ref_eval_mlm_accuracy", 0.0)),
        "best_trial_number": int(best_trial.number),
        "best_params": best_trial.params,
        "portable_overrides": best_portable,
        "best_run_path": str(best_run_path) if best_run_path else None,
        "canonical_pretrained_dir": str(BASE_RESULTS_DIR / SPLIT_TYPE / EXPORT_RUN_ID / "pretraining"),
    }
    best_config_path = OPTUNA_DIR / "best_pretrain_config.json"
    with open(best_config_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved best pre-training summary to: {best_config_path}")


if __name__ == "__main__":
    main()
