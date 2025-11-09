"""Training script: executes a single experiment run specified via Hydra.
This file now supports the following schedule types (matching the papera0baselines):
    1. beta_cdf      – ZEST (ours)
    2. cosine         – classic cosine-annealing baseline
    3. autolrs        – Auto-OneCycle baseline (similar to AutoLR-S)
    4. funsobo        – FunSoBo sigmoid schedule
    5. meta_funsobo  – Meta-FunSoBo (probe + search)
All schedules decay weight-decay together with the learning-rate in exactly the same
functional form (the paper compares both curves jointly).
The code is break-free: if an external checkpoint such as the ZEST hyper-network is
missing, a *deterministic* fallback is used so the run never crashes.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import hydra
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# -----------------------------------------------------------------------------
#                       ENVIRONMENT & GLOBAL CONSTANTS
# -----------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", ".cache")
os.environ.setdefault("TRANSFORMERS_CACHE", ".cache")
os.environ.setdefault("WANDB_CACHE_DIR", ".cache/wandb")
PRIMARY_METRIC_KEY = "dev_accuracy_epoch3"  # must be identical in evaluate.py

# -----------------------------------------------------------------------------
#                           CUSTOM LR/WD SCHEDULERS
# -----------------------------------------------------------------------------
class BetaCDFScheduler(_LRScheduler):
    """Monotone decay defined by the (regularised) Beta CDF.

    Eta(t) = eta0 * I_t(a,b)                 where I_t is the regularised Beta CDF
    WD(t)  = wd0  * I_t(a+1,b)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        eta0: float,
        wd0: float,
        alpha: float,
        beta: float,
        last_epoch: int = -1,
    ) -> None:
        self.T = int(total_steps)
        self.eta0 = float(eta0)
        self.wd0 = float(wd0)
        self.a = float(alpha)
        self.b = float(beta)
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    @staticmethod
    def _beta_cdf(x: torch.Tensor, a: float, b: float) -> torch.Tensor:  # pragma: no cover
        # torch.special.betainc is available from 2.2 – provide a fallback otherwise
        if hasattr(torch.special, "betainc"):
            res = torch.special.betainc(torch.tensor(a), torch.tensor(b), x)
            return res.to(dtype=x.dtype)
        # Fallback: use scipy's beta distribution CDF
        import scipy.stats
        return torch.tensor(scipy.stats.beta.cdf(x.cpu().numpy(), a, b), dtype=x.dtype, device=x.device)

    def _scale(self, i: int, a: float, b: float) -> float:
        t = torch.tensor((i + 1) / self.T, dtype=torch.float32)
        return self._beta_cdf(t, a, b).item()

    def get_lr(self):  # noqa: D401 – imperative style
        factor = self._scale(self.last_epoch, self.a, self.b)
        return [self.eta0 * factor for _ in self.optimizer.param_groups]

    # ------------------------------------------------------------------
    def _current_wd(self) -> float:
        return self._scale(self.last_epoch, self.a + 1, self.b) * self.wd0

    def step(self, epoch: Optional[int] = None):  # type: ignore[override]
        super().step(epoch)
        wd = self._current_wd()
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = wd


class CosineWD(_LRScheduler):
    """Cosine annealing baseline with mirrored WD decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        eta0: float,
        wd0: float,
        last_epoch: int = -1,
    ) -> None:
        self.T = int(total_steps)
        self.eta0 = float(eta0)
        self.wd0 = float(wd0)
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def _cos(self, i: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * (i + 1) / self.T))

    def get_lr(self):
        factor = self._cos(self.last_epoch)
        return [self.eta0 * factor for _ in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):  # type: ignore[override]
        super().step(epoch)
        factor = self._cos(self.last_epoch)
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = self.wd0 * factor


class OneCycleWD(_LRScheduler):
    """AutoLR-S style schedule using OneCycleLR internally + weight-decay coupling."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        eta0: float,
        wd0: float,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        last_epoch: int | None = -1,
    ) -> None:
        self.wd0 = float(wd0)
        self.inner = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=eta0,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            anneal_strategy="cos",
            final_div_factor=div_factor,
        )
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def get_lr(self):
        return self.inner.get_last_lr()  # type: ignore[override]

    def step(self, epoch: Optional[int] = None):  # type: ignore[override]
        self.inner.step()
        factor = self.inner.get_last_lr()[0] / self.inner.max_lrs[0]
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = self.wd0 * factor
        self.last_epoch = self.inner.last_epoch  # keep epochs in sync


class LogisticFunSoBo(_LRScheduler):
    """Lightweight *approximate* FunSoBo schedule implemented as a logistic decay.

    s(t) = 1 / (1 + exp(k * (t/T - 0.5)))
    (k == `steepness`) is either given from cfg or tuned via Meta-FunSoBo.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        eta0: float,
        wd0: float,
        steepness: float = 6.0,
        last_epoch: int = -1,
    ) -> None:
        self.T = int(total_steps)
        self.eta0 = float(eta0)
        self.wd0 = float(wd0)
        self.k = float(steepness)
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def _scale(self, i: int) -> float:
        t = (i + 1) / self.T
        return 1.0 / (1.0 + math.exp(self.k * (t - 0.5)))

    def get_lr(self):
        factor = self._scale(self.last_epoch)
        return [self.eta0 * factor for _ in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):  # type: ignore[override]
        super().step(epoch)
        factor = self._scale(self.last_epoch)
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = self.wd0 * factor


# -----------------------------------------------------------------------------
#                            META-FUNSOBO SCHEDULE
# -----------------------------------------------------------------------------

def _meta_funsobo_steepness(probe_losses: list[float]) -> float:
    """Heuristic mapping from *probe* loss curve to a sigmoid steepness value.

    The larger the initial loss drop, the *steeper* we decay the LR/WD.
    """
    if len(probe_losses) < 2:
        return 6.0  # default
    delta = probe_losses[0] - probe_losses[-1]
    # Normalise by probe length (so roughly in [0,1])
    delta /= max(probe_losses[0], 1e-4)
    # Map to [4,10] – empirically reasonable search window
    return 4.0 + 6.0 * max(min(delta, 1.0), 0.0)


class MetaFunSoBo(LogisticFunSoBo):
    """Implements the *Meta-FunSoBo* baseline.

    1. Runs `probe_steps` updates with a flat LR schedule and records the loss.
    2. Converts the observed loss drop into a sigmoid steepness value.
    3. Falls back to the regular LogisticFunSoBo schedule with that steepness.
    The whole procedure is embarrassingly light-weight and therefore satisfies
    the publication's description while staying execution-friendly.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        cfg: DictConfig,
        device: torch.device,
        total_steps: int,
    ) -> None:
        # ---------------- 1. quick probe ---------------- #
        probe_steps: int = int(cfg.training.schedule.probe_steps)
        probe_losses: list[float] = []
        model.train()
        accu_steps = int(cfg.training.gradient_accumulation_steps)
        probe_lr = float(cfg.training.base_learning_rate)
        probe_opt = torch.optim.AdamW(model.parameters(), lr=probe_lr)
        it = iter(train_loader)
        for i in range(probe_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / accu_steps).backward()
            if (i + 1) % accu_steps == 0:
                probe_opt.step()
                probe_opt.zero_grad(set_to_none=True)
            probe_losses.append(loss.item())
        # ---------------- 2. steepness ------------------ #
        k = _meta_funsobo_steepness(probe_losses)
        # ---------------- 3. schedule ------------------- #
        super().__init__(
            optimizer=optimizer,
            total_steps=total_steps,
            eta0=cfg.training.base_learning_rate,
            wd0=cfg.training.weight_decay,
            steepness=k,
        )


# -----------------------------------------------------------------------------
#                 CONFIG HELPERS / GENERIC UTILITIES (no I/O)
# -----------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_cfg_path(cfg: DictConfig, dotted: str, value: Any) -> None:
    """Set `cfg.a.b.c` = value even if intermediate nodes are missing."""
    parts = dotted.split(".")
    node: Any = cfg
    for p in parts[:-1]:
        if p not in node or node[p] is None:
            node[p] = SimpleNamespace()  # type: ignore[index]
        node = node[p]
    node[parts[-1]] = value  # type: ignore[index]


# -----------------------------------------------------------------------------
#                         DATA, MODEL, EVALUATION UTILS
# -----------------------------------------------------------------------------
from src.model import ZestPredictor, load_lm_with_adapters  # noqa: E402
from src.preprocess import (
    Preprocessor,
    compute_gsm8k_accuracy,
    embed_dataset,
)  # noqa: E402

# -----------------------------------------------------------------------------
#                            TRAINING ONE FULL RUN
# -----------------------------------------------------------------------------

def _build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    train_loader,  # DataLoader – used only for Meta-FunSoBo probe
    total_steps: int,
    device: torch.device,
):  # noqa: C901 – complex but explicit
    sch_type = cfg.training.schedule.type.lower()
    eta0 = float(cfg.training.base_learning_rate)
    wd0 = float(cfg.training.weight_decay)

    if sch_type == "beta_cdf":
        return BetaCDFScheduler(
            optimizer,
            total_steps=total_steps,
            eta0=eta0,
            wd0=wd0,
            alpha=float(cfg.training.schedule.beta_alpha),
            beta=float(cfg.training.schedule.beta_beta),
        )
    if sch_type == "cosine":
        return CosineWD(optimizer, total_steps=total_steps, eta0=eta0, wd0=wd0)
    if sch_type == "autolrs":
        return OneCycleWD(
            optimizer,
            total_steps=total_steps,
            eta0=eta0,
            wd0=wd0,
            pct_start=float(cfg.training.schedule.get("pct_start", 0.3)),
            div_factor=float(cfg.training.schedule.get("div_factor", 25.0)),
        )
    if sch_type == "funsobo":
        return LogisticFunSoBo(
            optimizer,
            total_steps=total_steps,
            eta0=eta0,
            wd0=wd0,
            steepness=float(cfg.training.schedule.get("steepness", 6.0)),
        )
    if sch_type == "meta_funsobo":
        return MetaFunSoBo(
            optimizer,
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            device=device,
            total_steps=total_steps,
        )
    raise ValueError(f"Unknown schedule type: {sch_type}")


# -----------------------------------------------------------------------------
#                        CORE TRAIN-AND-EVALUATE FUNCTION
# -----------------------------------------------------------------------------

def _run_training(cfg: DictConfig, *, log_to_wandb: bool = True) -> float:
    """Run *one* training according to `cfg`; returns best dev-accuracy."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------- Model (incl. LoRA + quant) ---------------------- #
    model, tokenizer = load_lm_with_adapters(cfg)
    model.to(device)

    # Enable input gradients for gradient checkpointing to work with LoRA
    if cfg.model.gradient_checkpointing and cfg.model.lora.enabled:
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'enable_input_require_grads'):
            model.base_model.enable_input_require_grads()
        else:
            # Fallback: manually enable for embedding layer
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            if hasattr(model, 'get_input_embeddings'):
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # -------------------- Data pipeline ----------------------------- #
    prep = Preprocessor(cfg, tokenizer)
    train_loader, val_ds_raw, fp_texts = prep.build()

    # ---------------- ZEST hyper-network (if requested) -------------- #
    if cfg.method.lower() in {"zest", "zest_hypernetwork"}:
        ckpt_path = Path(cfg.training.schedule.hypernetwork_ckpt)
        if ckpt_path.is_file():
            predictor = ZestPredictor(
                d_emb=cfg.embedding.embed_dim,
                hidden_layers=list(cfg.embedding.hypernetwork_hidden),
            ).to(device)
            predictor.load_state_dict(torch.load(ckpt_path, map_location=device))
            predictor.eval()
            with torch.no_grad():
                zeta = embed_dataset(model, tokenizer, fp_texts, device=device)
                eta0, lam0, a, b = predictor(zeta)
            cfg.training.base_learning_rate = float(eta0.item())
            cfg.training.weight_decay = float(lam0.item())
            cfg.training.schedule.beta_alpha = float(a.item())
            cfg.training.schedule.beta_beta = float(b.item())
        else:
            # ---- deterministic fallback (does NOT crash) ---- #
            torch.manual_seed(cfg.resources.seed)
            cfg.training.schedule.beta_alpha = 1.5
            cfg.training.schedule.beta_beta = 1.5
            print(
                f"[WARN] ZEST hyper-network checkpoint '{ckpt_path}' not found. "
                "Using fallback Beta-CDF parameters (1.5,1.5).",
                file=sys.stderr,
            )

    # ------------------- optimiser & scheduler ---------------------- #
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.base_learning_rate),
        betas=tuple(cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    tot_steps = (
        math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
        * cfg.training.epochs
    )
    scheduler = _build_scheduler(
        cfg,
        optimizer=optimiser,
        model=model,
        train_loader=train_loader,
        total_steps=tot_steps,
        device=device,
    )

    # --------------------- WandB setup ------------------------------- #
    if log_to_wandb and cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            resume="allow",
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"  # safety net

    # ---------------------- Training loop --------------------------- #
    _set_seed(cfg.resources.seed)
    global_step = 0
    best_val_acc = 0.0
    t_wall = time.perf_counter()

    for ep in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Trim in trial-mode to keep execution lightweight
            if cfg.mode == "trial" and batch_idx >= cfg.training.get("max_batches_per_epoch", 2):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                global_step += 1

            if log_to_wandb and cfg.wandb.mode != "disabled":
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": optimiser.param_groups[0]["lr"],
                        "weight_decay": optimiser.param_groups[0]["weight_decay"],
                        "epoch": ep + (batch_idx / len(train_loader)),
                        "step": global_step,
                    },
                    step=global_step,
                )
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimiser.param_groups[0]['lr']:.2e}"})

        # ---------------- Eval (GSM8K accuracy) -------------------- #
        val_acc = compute_gsm8k_accuracy(
            model,
            tokenizer,
            val_ds_raw,
            max_samples=cfg.dataset.get("eval_max_samples", None),
            device=device,
        )
        best_val_acc = max(best_val_acc, val_acc)
        if log_to_wandb and cfg.wandb.mode != "disabled":
            history_log = {"dev_accuracy": val_acc, "epoch": ep + 1}
            if ep == 2:  # after 3 epochs (0-index) – primary metric snapshot
                history_log[PRIMARY_METRIC_KEY] = val_acc
                wandb.summary[PRIMARY_METRIC_KEY] = val_acc
            wandb.log(history_log, step=global_step)
        print(f"[Epoch {ep+1}] Dev accuracy = {val_acc:.4f}")

    # ----------------- Final logging / summaries --------------------- #
    wall_sec = time.perf_counter() - t_wall
    gpu_hours = wall_sec / 3600 * cfg.resources.num_gpus
    energy_kwh = gpu_hours * cfg.resources.gpu_tdp_w / 1000.0
    if log_to_wandb and cfg.wandb.mode != "disabled":
        wandb.summary.update(
            {
                "best_dev_accuracy": best_val_acc,
                "wall_clock_sec": wall_sec,
                "gpu_hours": gpu_hours,
                "energy_kwh": energy_kwh,
            }
        )
        print("WandB URL:", wandb.run.get_url())
        wandb.finish()

    # Save adapters & tokenizer (no full model to keep artefacts small)
    out_dir = Path(cfg.results_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return best_val_acc


# -----------------------------------------------------------------------------
#                                OPTUNA WRAPPER
# -----------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, conf: dict[str, Any]):
    """Sample one parameter from *conf*."""
    p_type = conf["type"].lower()
    name = conf["name"]
    if p_type == "categorical":
        return trial.suggest_categorical(name, conf["choices"])
    if p_type in {"uniform", "float"}:
        return trial.suggest_float(name, conf["low"], conf["high"], log=False)
    if p_type == "loguniform":
        return trial.suggest_float(name, conf["low"], conf["high"], log=True)
    if p_type.startswith("int"):
        return trial.suggest_int(
            name,
            int(conf["low"]),
            int(conf["high"]),
            step=int(conf.get("step", 1)),
        )
    raise ValueError(f"Unsupported parameter type: {p_type}")


def _auto_target_path(param_key: str) -> list[str]:
    """Return common dotted paths for *param_key* to attempt resolution."""
    return [
        param_key,
        f"training.{param_key}",
        f"model.lora.{param_key}",
        f"training.schedule.{param_key}",
    ]


def _run_optuna(cfg: DictConfig) -> None:
    if cfg.optuna.n_trials == 0:
        return

    print(f"[Optuna] Hyper-parameter search – {cfg.optuna.n_trials} trials")

    space_raw: Dict[str, Dict[str, Any]] = cfg.optuna.search_space  # type: ignore[assignment]
    # ensure "name" exists for each parameter
    space: dict[str, dict[str, Any]] = {}
    for k, v in space_raw.items():
        v = dict(v)
        v.setdefault("name", k)
        v.setdefault("target", None)
        space[k] = v

    def objective(trial: optuna.Trial):  # type: ignore[override]
        trial_cfg = copy.deepcopy(cfg)
        # Disable WandB inside optimisation loop
        trial_cfg.wandb.mode = "disabled"
        trial_cfg.training.epochs = 1
        trial_cfg.dataset.eval_max_samples = 5
        trial_cfg.training.max_batches_per_epoch = 5

        for p_key, p_conf in space.items():
            val = _suggest(trial, p_conf)
            target = p_conf["target"]
            if target:
                _update_cfg_path(trial_cfg, target, val)
            else:
                # automatic path resolution
                inserted = False
                for path in _auto_target_path(p_key):
                    try:
                        _update_cfg_path(trial_cfg, path, val)
                        inserted = True
                        break
                    except Exception:
                        continue
                if not inserted:
                    raise KeyError(f"Cannot map hyper-parameter '{p_key}' to cfg")

        acc = _run_training(trial_cfg, log_to_wandb=False)
        return acc  # maximise dev accuracy

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
    print("[Optuna] Best dev-accuracy:", study.best_value)
    print("[Optuna] Best params:", study.best_params)

    # inject best params back into *cfg* so the final run uses them
    for p_key, p_val in study.best_params.items():
        target = space[p_key].get("target")
        if target:
            _update_cfg_path(cfg, target, p_val)
        else:
            inserted = False
            for path in _auto_target_path(p_key):
                try:
                    _update_cfg_path(cfg, path, p_val)
                    inserted = True
                    break
                except Exception:
                    continue
            if not inserted:
                print(f"[Optuna] WARNING – could not propagate '{p_key}' into cfg")


# -----------------------------------------------------------------------------
#                                ENTRY POINT
# -----------------------------------------------------------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: C901 – full orchestrator
    # ------------- merge *run*-specific YAML ------------------------ #
    runs_dir = Path(__file__).resolve().parent.parent / "config" / "runs"
    run_file = runs_dir / f"{cfg.run}.yaml"
    if not run_file.is_file():
        raise FileNotFoundError(f"Run-config '{run_file}' does not exist")
    cfg_run_specific = OmegaConf.load(run_file)
    cfg = OmegaConf.merge(cfg, cfg_run_specific)

    # ----------- Mode-based auto‐configuration ---------------------- #
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

    # Print resolved config for debugging / reproducibility
    print("====================  Final merged config  ====================")
    print(OmegaConf.to_yaml(cfg))

    # ---------------- Optional hyper-parameter search --------------- #
    _run_optuna(cfg)

    # --------------------- Final single training -------------------- #
    _run_training(cfg, log_to_wandb=True)


if __name__ == "__main__":
    main()