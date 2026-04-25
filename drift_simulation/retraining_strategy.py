"""
Task 8: Intelligent Retraining Strategy
Three strategies compared:
  1. Threshold-based  — retrain when recall drops below threshold
  2. Periodic         — retrain every N days regardless
  3. Hybrid           — periodic baseline + immediate trigger on severe drift

Compares: stability, compute cost, performance improvement
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Strategy definitions ──────────────────────────────────────────────────────
STRATEGIES = {
    "threshold_based": {
        "recall_threshold": 0.80,
        "psi_threshold":    0.15,
        "description":      "Retrain when recall < 0.80 OR PSI > 0.15",
    },
    "periodic": {
        "retrain_every_days": 14,
        "description":        "Retrain every 14 days",
    },
    "hybrid": {
        "recall_threshold":   0.82,
        "psi_threshold":      0.20,
        "retrain_every_days": 30,
        "description":        "Periodic (30d) + immediate trigger on severe drift",
    },
}


def simulate_model_decay(n_periods: int = 30, base_recall: float = 0.88,
                          decay_rate: float = 0.008, noise: float = 0.015,
                          random_state: int = 42) -> list:
    """
    Simulate how model recall degrades over time without retraining.
    Returns list of recall values per day.
    """
    rng = np.random.default_rng(random_state)
    recalls = []
    current = base_recall
    for i in range(n_periods):
        # Natural decay + noise
        current = current - decay_rate + rng.normal(0, noise)
        # Occasional sudden drop (new fraud pattern)
        if i in [10, 22]:
            current -= 0.05
        current = max(0.50, min(0.99, current))
        recalls.append(round(current, 4))
    return recalls


def simulate_psi_trend(n_periods: int = 30, random_state: int = 42) -> list:
    """Simulate PSI growing over time due to data drift."""
    rng = np.random.default_rng(random_state)
    psi_vals = []
    current = 0.02
    for i in range(n_periods):
        current += rng.uniform(0.003, 0.015)
        if i in [10, 22]:
            current += 0.08  # sudden spike
        current = min(current, 0.5)
        psi_vals.append(round(current, 4))
    return psi_vals


def retrain_effect(current_recall: float, base_recall: float = 0.88) -> float:
    """After retraining, recall returns to near-baseline with small random variation."""
    return round(min(0.99, base_recall + np.random.normal(0, 0.01)), 4)


def retrain_effect_psi() -> float:
    """After retraining, PSI resets to near zero."""
    return round(max(0.01, np.random.normal(0.03, 0.01)), 4)


def run_threshold_strategy(recalls: list, psi_vals: list, config: dict) -> dict:
    """Trigger retraining whenever recall or PSI crosses threshold."""
    recall_thresh = config["recall_threshold"]
    psi_thresh    = config["psi_threshold"]

    retrain_days   = []
    actual_recalls = list(recalls)
    actual_psi     = list(psi_vals)
    compute_cost   = 0.0  # arbitrary units (1.0 per retrain)

    for day in range(len(recalls)):
        if actual_recalls[day] < recall_thresh or actual_psi[day] > psi_thresh:
            retrain_days.append(day)
            compute_cost += 1.0
            # Apply retrain effect to future days
            new_recall = retrain_effect(actual_recalls[day])
            new_psi    = retrain_effect_psi()
            for future in range(day + 1, len(recalls)):
                decay = (future - day) * 0.008
                actual_recalls[future] = max(0.5, new_recall - decay + np.random.normal(0, 0.01))
                actual_psi[future] = min(0.5, new_psi + (future - day) * 0.008)

    return {
        "strategy": "threshold_based",
        "recalls":   actual_recalls,
        "psi_vals":  actual_psi,
        "retrain_days": retrain_days,
        "n_retrains": len(retrain_days),
        "avg_recall": round(np.mean(actual_recalls), 4),
        "min_recall": round(np.min(actual_recalls), 4),
        "compute_cost": round(compute_cost, 1),
        "stability": round(1.0 - np.std(actual_recalls), 4),
    }


def run_periodic_strategy(recalls: list, psi_vals: list, config: dict) -> dict:
    """Retrain every N days regardless of performance."""
    interval = config["retrain_every_days"]

    retrain_days   = []
    actual_recalls = list(recalls)
    actual_psi     = list(psi_vals)
    compute_cost   = 0.0

    for day in range(0, len(recalls), interval):
        retrain_days.append(day)
        compute_cost += 1.0
        new_recall = retrain_effect(actual_recalls[day])
        new_psi    = retrain_effect_psi()
        for future in range(day + 1, min(day + interval + 1, len(recalls))):
            decay = (future - day) * 0.008
            actual_recalls[future] = max(0.5, new_recall - decay + np.random.normal(0, 0.01))
            actual_psi[future] = min(0.5, new_psi + (future - day) * 0.008)

    return {
        "strategy": "periodic",
        "recalls":   actual_recalls,
        "psi_vals":  actual_psi,
        "retrain_days": retrain_days,
        "n_retrains": len(retrain_days),
        "avg_recall": round(np.mean(actual_recalls), 4),
        "min_recall": round(np.min(actual_recalls), 4),
        "compute_cost": round(compute_cost, 1),
        "stability": round(1.0 - np.std(actual_recalls), 4),
    }


def run_hybrid_strategy(recalls: list, psi_vals: list, config: dict) -> dict:
    """Periodic retraining + emergency trigger if recall/PSI threshold crossed."""
    interval      = config["retrain_every_days"]
    recall_thresh = config["recall_threshold"]
    psi_thresh    = config["psi_threshold"]

    retrain_days   = []
    actual_recalls = list(recalls)
    actual_psi     = list(psi_vals)
    compute_cost   = 0.0
    last_retrain   = -1

    for day in range(len(recalls)):
        trigger_periodic   = (day - last_retrain) >= interval
        trigger_emergency  = (actual_recalls[day] < recall_thresh or
                              actual_psi[day] > psi_thresh)

        if trigger_periodic or trigger_emergency:
            retrain_days.append(day)
            last_retrain = day
            # Emergency retrains cost slightly more (rushed)
            compute_cost += 1.5 if trigger_emergency and not trigger_periodic else 1.0
            new_recall = retrain_effect(actual_recalls[day])
            new_psi    = retrain_effect_psi()
            for future in range(day + 1, len(recalls)):
                decay = (future - day) * 0.008
                actual_recalls[future] = max(0.5, new_recall - decay + np.random.normal(0, 0.01))
                actual_psi[future] = min(0.5, new_psi + (future - day) * 0.008)

    return {
        "strategy": "hybrid",
        "recalls":   actual_recalls,
        "psi_vals":  actual_psi,
        "retrain_days": retrain_days,
        "n_retrains": len(retrain_days),
        "avg_recall": round(np.mean(actual_recalls), 4),
        "min_recall": round(np.min(actual_recalls), 4),
        "compute_cost": round(compute_cost, 1),
        "stability": round(1.0 - np.std(actual_recalls), 4),
    }


def plot_strategy_comparison(results: list, output_dir: str, n_periods: int):
    os.makedirs(output_dir, exist_ok=True)
    days = list(range(n_periods))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    colors = {"threshold_based": "blue", "periodic": "green", "hybrid": "red"}

    # ── Plot 1: Recall over time ──────────────────────────────────────────────
    for r in results:
        name   = r["strategy"]
        color  = colors[name]
        axes[0].plot(days, r["recalls"], label=f"{name} (avg={r['avg_recall']:.3f})",
                     color=color, linewidth=1.5)
        for rd in r["retrain_days"]:
            axes[0].axvline(rd, color=color, alpha=0.2, linestyle="--")

    axes[0].axhline(0.80, color="black", linestyle=":", label="Recall threshold (0.80)")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Fraud Recall Over Time — Strategy Comparison")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.5, 1.0)

    # ── Plot 2: PSI over time ─────────────────────────────────────────────────
    for r in results:
        name  = r["strategy"]
        color = colors[name]
        axes[1].plot(days, r["psi_vals"], label=name, color=color, linewidth=1.5)

    axes[1].axhline(0.10, color="orange", linestyle=":", label="Moderate drift (0.10)")
    axes[1].axhline(0.20, color="red",    linestyle=":", label="High drift (0.20)")
    axes[1].set_ylabel("PSI")
    axes[1].set_title("Data Drift (PSI) Over Time")
    axes[1].legend(fontsize=8)

    # ── Plot 3: Summary bar chart ─────────────────────────────────────────────
    names         = [r["strategy"]      for r in results]
    avg_recalls   = [r["avg_recall"]    for r in results]
    min_recalls   = [r["min_recall"]    for r in results]
    n_retrains    = [r["n_retrains"]    for r in results]
    compute_costs = [r["compute_cost"]  for r in results]
    stabilities   = [r["stability"]     for r in results]

    x = np.arange(len(names))
    w = 0.15
    axes[2].bar(x - 2*w, avg_recalls,   w, label="Avg Recall",    color="steelblue")
    axes[2].bar(x - 1*w, min_recalls,   w, label="Min Recall",    color="navy")
    axes[2].bar(x,        stabilities,  w, label="Stability",     color="green")
    axes[2].bar(x + 1*w, [c/max(compute_costs) for c in compute_costs],
                w, label="Compute Cost (norm)", color="orange")
    axes[2].bar(x + 2*w, [n/max(n_retrains) for n in n_retrains],
                w, label="Retrain Count (norm)", color="red")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_title("Strategy Comparison Summary")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(output_dir, "retraining_strategy_comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def compare_retraining_strategies(output_dir: str, report_path: str, n_periods: int = 60):
    print(f"\n{'='*60}")
    print("INTELLIGENT RETRAINING STRATEGY COMPARISON")
    print(f"{'='*60}")

    np.random.seed(42)
    base_recalls = simulate_model_decay(n_periods)
    base_psi     = simulate_psi_trend(n_periods)

    print(f"  Simulating {n_periods} days of model operation ...")
    print(f"  Base recall range: {min(base_recalls):.3f} – {max(base_recalls):.3f}")

    results = []
    for name, config in STRATEGIES.items():
        print(f"\n  Running strategy: {name}")
        print(f"    {config['description']}")
        np.random.seed(42)  # reset for fair comparison
        recalls_copy = list(base_recalls)
        psi_copy     = list(base_psi)

        if name == "threshold_based":
            r = run_threshold_strategy(recalls_copy, psi_copy, config)
        elif name == "periodic":
            r = run_periodic_strategy(recalls_copy, psi_copy, config)
        else:
            r = run_hybrid_strategy(recalls_copy, psi_copy, config)

        results.append(r)
        print(f"    Retrains     : {r['n_retrains']}")
        print(f"    Avg Recall   : {r['avg_recall']:.4f}")
        print(f"    Min Recall   : {r['min_recall']:.4f}")
        print(f"    Compute Cost : {r['compute_cost']:.1f} units")
        print(f"    Stability    : {r['stability']:.4f}")

    plot_strategy_comparison(results, output_dir, n_periods)

    # Best strategy selection
    # Score = avg_recall * stability / compute_cost
    scores = {
        r["strategy"]: (r["avg_recall"] * r["stability"]) / max(r["compute_cost"], 0.1)
        for r in results
    }
    best = max(scores, key=scores.get)
    print(f"\n  Best strategy (recall×stability/cost): {best}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "n_periods_simulated": n_periods,
        "strategies": {r["strategy"]: {
            "n_retrains": r["n_retrains"],
            "avg_recall": r["avg_recall"],
            "min_recall": r["min_recall"],
            "compute_cost": r["compute_cost"],
            "stability": r["stability"],
            "retrain_days": r["retrain_days"],
        } for r in results},
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "recommended_strategy": best,
        "recommendation_reason": (
            "Hybrid balances compute cost and performance "
            "by using periodic retraining as a baseline with "
            "emergency triggers for sudden drift events."
        ),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--report_path",  required=True)
    parser.add_argument("--n_periods",    type=int, default=60)
    args = parser.parse_args()
    compare_retraining_strategies(args.output_dir, args.report_path, args.n_periods)


if __name__ == "__main__":
    main()
