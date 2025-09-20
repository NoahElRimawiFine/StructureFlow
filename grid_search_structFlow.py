import os, itertools, json, time, copy, math, random
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from src.models.components.sf2m_ngm1 import SF2MNGM
from src.datamodules.grn_datamodule import TrajectoryStructureDataModule

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def as_hashable(cfg: dict):
    """Turn nested dict/list config into a stable string for logging/filenames."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))

def best_objective_from_model(model: SF2MNGM) -> float:
    """
    Objective to minimize.
    By default: best (minimum) total training loss observed.
    If you prefer the tracked ODE/SDE Wasserstein logged every 500 steps,
    you can compute/return that here instead (e.g., np.nanmean of last snapshot).
    """
    if len(model.loss_history) == 0:
        return float("inf")
    return float(np.min(model.loss_history))

def run_one(config: dict, base_data_path="data/", dataset_name="dyn-TF", results_dir="grid_runs"):

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    run_tag = str(abs(hash(as_hashable(config))))[:12]
    run_dir = Path(results_dir) / f"run_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility per run
    set_seed(config.get("seed", 42))

    # Build datamodule once per run
    datamodule = TrajectoryStructureDataModule(
        data_path=base_data_path,
        dataset=dataset_name,
        dataset_type="Synthetic",
    )

    # Instantiate the model with config values, keeping sane defaults for others
    model = SF2MNGM(
        datamodule=datamodule,
        T=config.get("T", 5),
        sigma=config.get("sigma", 1.0),
        dt=config.get("dt", 0.2),
        batch_size=config.get("batch_size", 164),
        alpha=config.get("alpha", 0.1),
        reg=config.get("reg", 0.0),
        dyn_hidden=config.get("dyn_hidden", 4),
        dyn_alpha=config.get("dyn_alpha", 0.1),
        correction_reg_strength=config.get("correction_reg_strength", 1e-3),
        n_steps=config.get("n_steps", 2000),      # keep modest for grid speed
        lr=config.get("lr", 1e-4),
        device=config.get("device", None),        # auto-detect
        GL_reg=config.get("GL_reg", 0.04),
        knockout_hidden=config.get("knockout_hidden", 100),
        score_hidden=config.get("score_hidden", [100, 100]),
        correction_hidden=config.get("correction_hidden", [64, 64]),
    )

    # Train
    t0 = time.time()
    model.train_model(skip_time=config.get("skip_time", None))
    elapsed = time.time() - t0

    # Compute an objective
    obj = best_objective_from_model(model)

    # Persist per-run summary
    summary = {
        "config": copy.deepcopy(config),
        "objective": float(obj),
        "elapsed_sec": float(elapsed),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return obj, summary


def grid_search():
    # --------- Define your grid here (edit freely) ---------
    grid = {
        # Train budget first: keep n_steps modest so the grid finishes
        "n_steps": [5_000, 10_000],
        # Optimizer & schedule-like knobs
        "lr": [5e-5, 1e-4, 3e-4],
        "batch_size": [64],
        # Loss mixing / modeling
        "alpha": [0.1, 0.2],
        "dyn_alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
        "dyn_hidden": [2,4,6,8],
        "sigma": [1.0],
        "dt": [0.2],
        # Regularization
        "reg": [0.0, 1e-5, 1e-6],
        "GL_reg": [0.02],
        # Architectures
        "score_hidden": [[128, 128]],
        "correction_hidden": [[64, 64]]
    }

    # Cartesian product
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total configs: {len(combos)}")

    results = []
    best = {"objective": float("inf"), "summary": None}

    for idx, values in enumerate(combos, 1):
        config = {k: v for k, v in zip(keys, values)}
        print(f"\n[{idx}/{len(combos)}] Running config:\n{json.dumps(config, indent=2)}")

        try:
            obj, summary = run_one(config=config)
        except Exception as e:
            print(f"Run failed with error: {e}")
            obj, summary = float("inf"), {"config": config, "error": str(e)}

        results.append({
            **summary.get("config", config),
            "objective": obj,
            "elapsed_sec": summary.get("elapsed_sec", float("nan")),
            "error": summary.get("error", None),
        })

        if obj < best["objective"]:
            best = {"objective": obj, "summary": summary}
            print(f"New best objective: {obj:.6f}")

        # Save rolling CSV so you can tail progress live
        df = pd.DataFrame(results)
        df.to_csv("grid_results.csv", index=False)

    print("\n=== Grid search complete ===")
    print("Best objective:", best["objective"])
    print("Best config:")
    print(json.dumps(best["summary"]["config"], indent=2))


if __name__ == "__main__":
    grid_search()