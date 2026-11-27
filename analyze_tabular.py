"""
analyze_tabular.py - recompute convergence metrics for ALL saved tabular runs from their
results/tabular_*_log.csv files (no retraining needed) and print one summary table.

Metrics (per run), all computed from the greedy-policy checkpoints stored in the log:
  conv_end    : first episode E such that, from E until the END of training, EVERY checkpoint has
                success >= 95% and mean rolls <= 1.10 * BFS optimum.   (None = not converged)
  opt_end     : same, but mean rolls must EQUAL the BFS optimum.      (None = never stayed optimal)
  opt_first   : first checkpoint at which the greedy policy was exactly optimal (even briefly).
  opt_lost    : number of checkpoints AFTER opt_first where the policy was NOT optimal.
                > 0 means the greedy policy was optimal once and then got worse (unstable).
Why "until the end"? A policy that is good at episode 40 but worse again at episode 200 has not
really converged. The shorter 50-episode window used inside tabular.py can miss this.

Usage:  python analyze_tabular.py
Outputs: results/tabular_run_table.csv (one row per run), results/tabular_aggregate.csv
Runs saved with a tag (e.g. _extra) form their own group; 'main' = no tag; pilot files are skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from env import bfs_min_rolls

RESULTS = Path(__file__).parent / "results"
PAT = re.compile(r"tabular_(qlearning|sarsa)_(N\d+)_seed(\d+)(.*)_log\.csv$")


def stable_to_end(episodes: np.ndarray, ok: np.ndarray):
    """First checkpoint episode from which every later checkpoint is ok (None if the last one is not)."""
    if len(ok) == 0 or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return int(episodes[i])


def analyze_run(path: Path, algo: str, board: str, seed: int, tag: str) -> dict:
    log = pd.read_csv(path)
    ev = log.dropna(subset=["eval_success"]).reset_index(drop=True)
    bfs = bfs_min_rolls(__import__("env").BOARDS[board])
    eps = ev["episode"].to_numpy()
    succ, moves = ev["eval_success"].to_numpy(), ev["eval_moves"].to_numpy()
    loose = (succ >= 0.95) & (moves <= 1.10 * bfs)
    strict = (succ >= 0.95) & (moves <= bfs + 1e-9)
    first_idx = int(np.argmax(strict)) if strict.any() else None
    last100 = log.tail(100)
    return {
        "board": board, "algo": algo, "seed": seed, "group": tag.strip("_") or "main", "bfs": bfs,
        "episodes_run": int(log["episode"].max()),
        "conv_end": stable_to_end(eps, loose),
        "opt_end": stable_to_end(eps, strict),
        "opt_first": None if first_idx is None else int(eps[first_idx]),
        "opt_lost": 0 if first_idx is None else int((~strict[first_idx:]).sum()),
        "final_moves": float(moves[-1]), "final_return": float(ev["eval_return"].iloc[-1]),
        "train_moves_last100": float(last100["train_moves"].mean()),
        "train_return_last100": float(last100["train_return"].mean()),
    }


def describe(series: pd.Series):
    v = series.dropna().astype(float)
    if len(v) == 0:
        return "-", "-", "-"
    sd = f"{v.std(ddof=1):.1f}" if len(v) > 1 else "-"
    return f"{v.mean():.1f}", sd, f"{v.median():.1f} [{v.min():.0f}-{v.max():.0f}]"


def main():
    rows = []
    for p in sorted(RESULTS.glob("tabular_*_log.csv")):
        m = PAT.match(p.name)
        if not m or "pilot" in p.name:
            continue
        algo, board, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        rows.append(analyze_run(p, algo, board, seed, tag))
    if not rows:
        print("No tabular logs found in results/. Run tabular.py first.")
        return
    runs = pd.DataFrame(rows).sort_values(["board", "algo", "group", "seed"])
    runs.to_csv(RESULTS / "tabular_run_table.csv", index=False)

    agg_rows = []
    for (board, algo), g in runs.groupby(["board", "algo"]):
        parts = [("main", g[g.group == "main"])] if (g.group == "main").any() else []
        if (g.group != "main").any():
            parts.append(("extra" if g.group.nunique() == 2 else "other", g[g.group != "main"]))
        if len(parts) > 1:
            parts.append(("all", g))
        for name, gg in parts:
            cm, cs, cmed = describe(gg["conv_end"])
            om, os_, omed = describe(gg["opt_end"])
            agg_rows.append({
                "board": board, "algo": algo, "seeds": name, "n": len(gg),
                "conv_end mean": cm, "sd": cs, "median [min-max]": cmed,
                "not_conv": int(gg["conv_end"].isna().sum()),
                "opt_end mean": om, "sd ": os_, "median [min-max] ": omed,
                "never_opt": int(gg["opt_end"].isna().sum()),
                "unstable_runs": int((gg["opt_lost"] > 0).sum()),
                "final_moves": f"{gg['final_moves'].mean():.2f}",
                "final_return": f"{gg['final_return'].mean():.1f}",
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(RESULTS / "tabular_aggregate.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg.to_string(index=False))
    print("\nconv_end / opt_end are in EPISODES; mean and sd only over runs that reached them; "
          "not_conv / never_opt count the runs that did not.")
    print("unstable_runs = runs whose greedy policy was exactly optimal at some checkpoint and later got worse.")
    print("Saved results/tabular_run_table.csv and results/tabular_aggregate.csv")


if __name__ == "__main__":
    main()
