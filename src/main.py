"""Experiment orchestrator – launches *one* run via a subprocess call to src.train.
This wrapper exists so Hydra handles *global* CLI args (mode, results_dir, run) only
once.  The heavy-weight config merge happens inside src.train again for safety.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # Validate mode & apply quick overrides for trial/full execution
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.dataset.eval_max_samples = 2
        cfg.training.max_batches_per_epoch = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    print("================  Launcher config  ================")
    print(OmegaConf.to_yaml(cfg))

    # Build subprocess call – disable nested Hydra run directories to keep paths clean
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if Path.cwd().name == "src":
        print("[WARN] Execute the launcher from the project root: `python -m src.main …`.")
    main()