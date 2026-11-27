"""
unified_convergence.py - ONE convergence rule for all five algorithms, computed from saved logs.
No training happens here; it only re-reads results/*.csv written by tabular.py and sb3_train.py.

RULE (strict, same for tabular and deep RL):
  A checkpoint is "good" if greedy success >= 95% AND mean greedy rolls <= 1.10 * BFS optimum.
  Converged at checkpoint E if E and EVERY LATER checkpoint up to the end of training are good.
  If the last checkpoint is not good, the run is "not converged".
  "Exactly optimal" = same rule with mean rolls <= BFS optimum.
  Cost of convergence is reported in training EPISODES and in ENVIRONMENT STEPS (rolls) so the
  tabular and deep methods can be put on the same axis.

Run from the project folder:  python unified_convergence.py
Writes results/unified_convergence_per_run.csv and results/unified_convergence_summary.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from env import BOARDS, bfs_min_rolls

RES = Path(__file__).parent / "results"
SUCC, SLACK = 0.95, 1.10


def stable_start(ok: np.ndarray):
    """Index of the first checkpoint from which all later checkpoints are ok; None if the last is not ok."""
    if len(ok) == 0 or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i


def analyse(ev: pd.DataFrame, bfs: int):
    """ev needs columns: episodes, steps, success, moves (one row per greedy checkpoint, in order)."""
    loose = ((ev.success >= SUCC) & (ev.moves <= SLACK * bfs)).to_numpy()
    exact = ((ev.success >= SUCC) & (ev.moves <= bfs + 1e-9)).to_numpy()
    out = {}
    for name, ok in (("conv", loose), ("opt", exact)):
        i = stable_start(ok)
        out[f"{name}_episode"] = None if i is None else int(ev.episodes.iloc[i])
        out[f"{name}_steps"] = None if i is None else int(ev.steps.iloc[i])
    out["converged"] = out["conv_episode"] is not None
    out["exactly_optimal"] = out["opt_episode"] is not None
    out["last_checkpoint_episode"] = int(ev.episodes.iloc[-1])
    return out


def tabular_rows():
    for f in sorted(RES.glob("tabular_*_log.csv")):
        m = re.fullmatch(r"tabular_(qlearning|sarsa)_(N\d+)_seed(\d+)(_extra)?_log", f.stem)
        if not m:                       # skips *_pilot_log.csv and anything unexpected
            continue
        algo, board, seed = m.group(1), m.group(2), int(m.group(3))
        log = pd.read_csv(f)
        log["steps"] = log["train_moves"].cumsum()
        ev = log.dropna(subset=["eval_success"]).rename(
            columns={"episode": "episodes", "eval_success": "success", "eval_moves": "moves"})
        yield algo, board, seed, f.name, ev[["episodes", "steps", "success", "moves"]], int(log.episode.max())


def sb3_rows():
    for f in sorted(RES.glob("sb3_*_evals.csv")):
        m = re.fullmatch(r"sb3_(dqn|ppo|a2c)_(N\d+)_seed(\d+)_evals", f.stem)
        if not m:
            continue
        algo, board, seed = m.group(1), m.group(2), int(m.group(3))
        d = pd.read_csv(f)
        ev = d.rename(columns={"timesteps": "steps", "eval_success": "success", "eval_moves": "moves"})
        yield algo, board, seed, f.name, ev[["episodes", "steps", "success", "moves"]], int(d.episodes.iloc[-1])


def fmt(series):
    s = series.dropna()
    if s.empty:
        return "-"
    return f"{s.median():.0f} [{s.min():.0f}-{s.max():.0f}]"


def main():
    rows = []
    for gen, group in ((tabular_rows(), "tabular"), (sb3_rows(), "deep")):
        for algo, board, seed, fname, ev, ep_total in gen:
            bfs = bfs_min_rolls(BOARDS[board])
            r = analyse(ev, bfs)
            rows.append({"group": group, "algo": algo, "board": board, "seed": seed, "file": fname,
                         "bfs": bfs, "episodes_total": ep_total, **r})
    per = pd.DataFrame(rows)
    per.to_csv(RES / "unified_convergence_per_run.csv", index=False)

    order = {"qlearning": 0, "sarsa": 1, "dqn": 2, "ppo": 3, "a2c": 4}
    summ = []
    for (board, algo), g in per.groupby(["board", "algo"]):
        summ.append({"board": board, "algo": algo, "runs": len(g),
                     "converged": int(g.converged.sum()), "exactly_optimal": int(g.exactly_optimal.sum()),
                     "conv_episodes_median[min-max]": fmt(g.conv_episode),
                     "conv_steps_median[min-max]": fmt(g.conv_steps),
                     "opt_episodes_median[min-max]": fmt(g.opt_episode)})
    sdf = pd.DataFrame(summ)
    sdf["_o"] = sdf.algo.map(order)
    sdf["_b"] = sdf.board.str[1:].astype(int)
    sdf = sdf.sort_values(["_b", "_o"]).drop(columns=["_o", "_b"])
    sdf.to_csv(RES / "unified_convergence_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sdf.to_string(index=False))
    print(f"\nRuns analysed: {len(per)} "
          f"(tabular {int((per.group == 'tabular').sum())}, deep {int((per.group == 'deep').sum())})")
    nc = per[~per.converged]
    if len(nc):
        print("\nNOT converged under the strict rule:")
        print(nc[["algo", "board", "seed", "episodes_total", "file"]].to_string(index=False))


if __name__ == "__main__":
    main()
