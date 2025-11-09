"""Independent evaluation script – fetches run histories from WandB and
produces per-run artefacts *plus* an aggregated comparison.
Usage:
    uv run python -m src.evaluate results_dir=out run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import sklearn.metrics as skm
import wandb

os.environ.setdefault("WANDB_CACHE_DIR", ".cache/wandb")
PRIMARY_METRIC_KEY = "dev_accuracy_epoch3"

# -----------------------------------------------------------------------------
#                           UTILITY HELPERS
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_learning_curves(history: pd.DataFrame, run_id: str, out_dir: Path) -> None:
    if history.empty:
        return
    plt.figure(figsize=(6, 4))
    for col in [c for c in history.columns if c.startswith("train_") or c.startswith("dev_")]:
        sns.lineplot(x=history.index, y=history[col], label=col)
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.title(f"Learning curves – {run_id}")
    plt.legend(frameon=False)
    file = out_dir / f"{run_id}_learning_curve.pdf"
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
    print(file)


def _plot_confusion_matrix(history: pd.DataFrame, run_id: str, out_dir: Path) -> None:
    if not {"true_label", "pred_label"}.issubset(history.columns):
        return
    cm = skm.confusion_matrix(history["true_label"], history["pred_label"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(f"Confusion – {run_id}")
    file = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
    print(file)


# -----------------------------------------------------------------------------
#                           PER-RUN EXPORT
# -----------------------------------------------------------------------------

def export_run(run: "wandb.apis.public.Run", out_root: Path) -> Dict[str, float]:
    _ensure_dir(out_root)

    # time-series history (all keys)
    history_df = run.history()
    if not history_df.empty:
        history_df.to_csv(out_root / "history.csv", index=False)

    summary = run.summary._json_dict
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_root / "config.json", "w") as f:
        json.dump(dict(run.config), f, indent=2)

    # Combined metrics.json as required by spec
    combo = {
        "summary": summary,
        "primary_metric": summary.get(PRIMARY_METRIC_KEY),
    }
    if not history_df.empty:
        combo["final_history_row"] = history_df.tail(1).to_dict(orient="records")[0]
    with open(out_root / "metrics.json", "w") as f:
        json.dump(combo, f, indent=2)
    print(out_root / "metrics.json")

    # Figures
    _plot_learning_curves(history_df, run.id, out_root)
    _plot_confusion_matrix(history_df, run.id, out_root)

    # Return only numeric summary entries
    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}


# -----------------------------------------------------------------------------
#                             CLI PARSING
# -----------------------------------------------------------------------------

def _parse_cli() -> Dict[str, str]:
    kv = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        kv[k] = v
    return kv


# -----------------------------------------------------------------------------
#                                   MAIN
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – imperative style
    cli = _parse_cli()
    if {"results_dir", "run_ids"}.difference(cli):
        raise SystemExit(
            "Usage: python -m src.evaluate results_dir=<dir> run_ids='[""run-1"",...]'",
        )

    results_dir = Path(cli["results_dir"])
    run_ids: List[str] = json.loads(cli["run_ids"])

    # -------------------------------- WandB API ------------------------------ #
    import yaml

    with open(Path("config") / "config.yaml", "r") as f:
        cfg_base = yaml.safe_load(f)
    entity = cfg_base["wandb"]["entity"]
    project = cfg_base["wandb"]["project"]

    api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = defaultdict(dict)
    numeric_summaries: Dict[str, Dict[str, float]] = {}

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        out_run_dir = _ensure_dir(results_dir / rid)
        metrics = export_run(run, out_run_dir)
        numeric_summaries[rid] = metrics
        for k, v in metrics.items():
            aggregated[k][rid] = v

    # ------------------- Aggregated JSON ------------------------------------ #
    comp_dir = _ensure_dir(results_dir / "comparison")

    proposed = {
        rid: numeric_summaries[rid][PRIMARY_METRIC_KEY]
        for rid in run_ids
        if any(t in rid.lower() for t in ("proposed", "zest"))
    }
    baseline = {
        rid: numeric_summaries[rid][PRIMARY_METRIC_KEY]
        for rid in run_ids
        if any(t in rid.lower() for t in ("baseline", "comparative"))
    }

    best_prop = max(proposed, key=proposed.get) if proposed else None
    best_base = max(baseline, key=baseline.get) if baseline else None
    gap_pct = None
    if best_prop and best_base:
        gap_pct = (
            (proposed[best_prop] - baseline[best_base]) / baseline[best_base] * 100.0
        )

    agg_json = {
        "primary_metric": "Dev-set accuracy after 3 epochs; equal accuracies broken by wall-clock time.",
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop, "value": proposed.get(best_prop) if best_prop else None},
        "best_baseline": {"run_id": best_base, "value": baseline.get(best_base) if best_base else None},
        "gap": gap_pct,
    }
    with open(comp_dir / "aggregated_metrics.json", "w") as f:
        json.dump(agg_json, f, indent=2)
    print(comp_dir / "aggregated_metrics.json")

    # ------------------------ Figures --------------------------------------- #
    # 1. Primary metric bar-chart
    plt.figure(figsize=(max(6, len(run_ids) * 1.4), 4))
    values = [aggregated[PRIMARY_METRIC_KEY][rid] for rid in run_ids]
    ax = sns.barplot(x=run_ids, y=values, palette="viridis")
    for idx, val in enumerate(values):
        ax.text(idx, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(PRIMARY_METRIC_KEY)
    plt.title("Dev accuracy after 3 epochs")
    bar_path = comp_dir / "comparison_dev_accuracy_epoch3_bar_chart.pdf"
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(bar_path)

    # 2. Wall-clock violin (if available)
    if "wall_clock_sec" in aggregated and all(r in aggregated["wall_clock_sec"] for r in run_ids):
        plt.figure(figsize=(6, 4))
        secs = [aggregated["wall_clock_sec"][rid] for rid in run_ids]
        sns.violinplot(data=secs)
        plt.ylabel("Seconds")
        plt.title("Wall-clock time distribution")
        violin_path = comp_dir / "comparison_wall_clock_violin.pdf"
        plt.tight_layout()
        plt.savefig(violin_path)
        plt.close()
        print(violin_path)

    # 3. Welch's t-test (proposed vs baseline)
    if len(proposed) > 1 and len(baseline) > 1:
        stat = st.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        print("Welch t-test p-value (proposed vs baseline):", stat.pvalue)


if __name__ == "__main__":
    main()