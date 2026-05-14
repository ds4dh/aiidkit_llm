import re
import numpy as np
from pathlib import Path
from datasets import Dataset


def scan_all_fups(data_dir: Path) -> list[int]:
    """
    Find all available follow-up folders (fup_XXXX) in the data directory
    """
    fups = []
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("fup_"):
            try:
                # Extract integer from "fup_0090" -> 90
                val = int(path.name.split("_")[-1])
                fups.append(val)
            except ValueError:
                continue  # skip fup_None or malformed folders
    
    return sorted(fups)


def prepare_dataset_fup_dict(dataset: Dataset, fup_list: list[int]):
    """
    Creates a dictionary of datasets for different follow-up periods.
    """
    out_dict = {"all": dataset}
    fup_array = np.array(dataset["fup"])
    for fup in fup_list:
        indices = np.where(fup_array == fup)[0]
        if len(indices) > 0:
            subset = dataset.select(indices)  # dataset view
            out_dict[f"fup_{fup:04d}"] = subset
            
    return out_dict


def find_best_checkpoint(base_dir: Path, task_key: str, horizon: int) -> Path:
    """
    ...
    """
    task_dir = base_dir / "finetuning" / task_key
    if not task_dir.exists(): raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    candidates = [p for p in task_dir.iterdir() if p.is_dir() and pattern.search(p.name)]
    if not candidates: raise FileNotFoundError(f"No run found for horizon {h_str} inside hrz() in {task_dir}")
    
    run_dir = candidates[0]
    checkpoint_dirs = sorted(
        list(run_dir.glob("checkpoint-*")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoint_dirs: raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    return checkpoint_dirs[0]


def extract_horizons_from_path(checkpoint_path: Path) -> list[int]:
    """
    ...
    """
    run_dir = checkpoint_path
    while "hrz(" not in run_dir.name and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    
    match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
    if not match:
        raise ValueError(f"Could not extract horizons from path: {run_dir.name}")
    
    return [int(h) for h in match.group(1).split("-")] 
