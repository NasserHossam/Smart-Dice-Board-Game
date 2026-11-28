"""
evaluate.py - builds the final tables from everything saved in results/.

Reads  : results/tabular/tabular_*_log.csv / _Q.npy   (Q-Learning, SARSA)
         results/deep_rl/sb3_*_summary.json / _evals.csv / _episodes.csv / _policy.npy   (DQN, PPO, A2C)
Writes : results/tables/final_table_per_run.csv, final_table_summary.csv, random_baseline.csv
Prints : the summary table (one block per board).

DEFINITIONS (put these in the report)
  final_*            greedy policy after training. Env + greedy policy are deterministic, so one
                     game stands for all 100 identical games; success is 0% or 100% per run, and the
                     success rate in the table is the share of SEEDS whose final policy reaches the goal.
  train_last100_*    mean over the last 100 TRAINING episodes (includes exploration noise).
  conv_episode       first training episode E such that from E until the END of training every
                     greedy checkpoint has success >= 95% and mean rolls <= 1.10 * BFS optimum.
                     Missing = not converged. (deep RL: also conv_timesteps)
  optimal_return     best possible undiscounted return, from dynamic programming (value iteration).
  state_agreement    share of reachable squares where the learned greedy roll lies on a BFS
                     shortest path (fewest rolls). Tabular: rows never updated are skipped.
  soft_last20        share of the last 20% of checkpoints that met the convergence criterion
                     (a softer stability number, useful when a run is 'not converged').

Usage:  python src/evaluate.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tabular import PAT as TAB_PAT
from analyze_tabular import analyze_run
from env import BOARDS, DEFAULT_REWARDS, bfs_min_rolls, next_square, optimal_roll_sets, run_random_policy

from paths import DEEP, TABLES, TABULAR
SB3_PAT = re.compile(r"sb3_(dqn|ppo|a2c)_(N\d+)_seed(\d+)(.*)_summary\.json$")
ALGO_ORDER = ["qlearning", "sarsa", "dqn", "ppo", "a2c"]
NAMES = {"qlearning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN", "ppo": "PPO", "a2c": "A2C"}


# --------------------------------------------------------------------------- reference values
def reachable_squares(board) -> set:
    seen, stack = {1}, [1]
    while stack:
        s = stack.pop()
        for roll in range(1, 7):
            nx = next_square(board, s, roll)
            if nx is not None and nx not in seen:
                seen.add(nx)
                stack.append(nx)
    return seen


def optimal_return(board, rewards: dict | None = None) -> float:
    """Best possible undiscounted return from square 1 (value iteration; assumes invalid-move
    reward <= 0, so an invalid move is never useful and is left out)."""
    r = {**DEFAULT_REWARDS, **(rewards or {})}
    n = board.n
    trans = {}
    for s in range(1, n):
        opts = []
        for roll in range(1, 7):
            t = s + roll
            if t > n:
                continue
            final = board.jump_map.get(t, t)
            if final == n:
                rew = r["goal"]
            elif t in board.ladders:
                rew = r["ladder"]
            elif t in board.snakes:
                rew = r["snake"]
            else:
                rew = r["normal"]
            opts.append((rew, final))
        trans[s] = opts
    v = np.zeros(n + 1)
    done = False
    for _ in range(10 * n + 100):
        new = v.copy()
        for s in range(1, n):
            new[s] = max(rew + v[f] for rew, f in trans[s])
        done = np.max(np.abs(new - v)) < 1e-9
        v = new
        if done:
            break
    return float(v[1]) if done else float("nan")


def policy_agreement(board, rolls_by_square: np.ndarray, skip: np.ndarray | None = None):
    """rolls_by_square[s-1] = roll (1..6) chosen at square s. Returns (share optimal, squares compared)."""
    sets = optimal_roll_sets(board)
    squares = [s for s in sorted(reachable_squares(board)) if s < board.n]
    ok = []
    for s in squares:
        if skip is not None and skip[s - 1]:
            continue
        ok.append(int(rolls_by_square[s - 1]) in sets[s])
    return (float(np.mean(ok)) if ok else np.nan), len(ok)


# --------------------------------------------------------------------------- per-run rows
def _train_windows(ret, moves, succ):
    ret, moves, succ = (np.asarray(x, dtype=float) for x in (ret, moves, succ))
    k = min(100, len(ret))
    return {
        "train_first100_moves": moves[:k].mean(), "train_first100_return": ret[:k].mean(),
        "train_first100_success": succ[:k].mean(),
        "train_last100_moves": moves[-k:].mean(), "train_last100_return": ret[-k:].mean(),
        "train_last100_success": succ[-k:].mean(),
    }


def _soft_last20(success, moves, bfs):
    ok = (np.asarray(success) >= 0.95) & (np.asarray(moves) <= 1.10 * bfs)
    k = max(1, int(round(0.2 * len(ok))))
    return float(ok[-k:].mean())


def tabular_row(path: Path) -> dict:
    algo, board, seed, tag = TAB_PAT.match(path.name).groups()
    seed = int(seed)
    base = analyze_run(path, algo, board, seed, tag)
    log = pd.read_csv(path)
    ev = log.dropna(subset=["eval_success"])
    b = BOARDS[board]
    stem = path.name[: -len("_log.csv")]
    Q = np.load(path.parent / f"{stem}_Q.npy")
    visited = (Q != 0).any(axis=1)
    agree, n_cmp = policy_agreement(b, Q.argmax(axis=1) + 1, skip=~visited)
    row = {
        "family": "tabular", "algo": algo, "board": board, "seed": seed, "group": base["group"],
        "bfs": base["bfs"], "optimal_return": optimal_return(b),
        "episodes_run": base["episodes_run"], "timesteps_run": np.nan,
        "final_success": float(ev["eval_success"].iloc[-1]), "final_moves": base["final_moves"],
        "final_return": base["final_return"],
        "conv_episode": base["conv_end"], "conv_timesteps": np.nan, "opt_episode": base["opt_end"],
        "soft_last20": _soft_last20(ev["eval_success"], ev["eval_moves"], base["bfs"]),
        "state_agreement": agree, "states_compared": n_cmp,
    }
    row.update(_train_windows(log["train_return"], log["train_moves"], log["train_success"]))
    return row


def deep_row(path: Path) -> dict:
    algo, board, seed, tag = SB3_PAT.match(path.name).groups()
    stem = path.name[: -len("_summary.json")]
    s = json.load(open(path))
    ev = pd.read_csv(path.parent / f"{stem}_evals.csv")
    eps = pd.read_csv(path.parent / f"{stem}_episodes.csv")
    b = BOARDS[board]
    policy = np.load(path.parent / f"{stem}_policy.npy")
    agree, n_cmp = policy_agreement(b, policy + 1)
    bfs = s["bfs_optimum"]
    ok_strict = (ev["eval_success"] >= 0.95) & (ev["eval_moves"] <= bfs + 1e-9)
    row = {
        "family": "deep", "algo": algo, "board": board, "seed": int(seed), "group": tag.strip("_") or "main",
        "bfs": bfs, "optimal_return": optimal_return(b),
        "episodes_run": s["episodes_run"], "timesteps_run": s["timesteps_run"],
        "final_success": s["final_eval_success"], "final_moves": s["final_eval_moves"],
        "final_return": s["final_eval_return"],
        "conv_episode": s["convergence_episode"] if s["converged"] else np.nan,
        "conv_timesteps": s["convergence_timesteps"] if s["converged"] else np.nan,
        "opt_episode": s["optimal_episode"] if s.get("optimal_episode") is not None else np.nan,
        "soft_last20": _soft_last20(ev["eval_success"], ev["eval_moves"], bfs),
        "state_agreement": agree, "states_compared": n_cmp, "stopped_by_time": s["stopped_by_time_limit"],
    }
    row.update(_train_windows(eps["return"], eps["moves"], eps["success"]))
    return row


def build_per_run() -> pd.DataFrame:
    rows = []
    for p in sorted(TABULAR.glob("tabular_*_log.csv")):
        if TAB_PAT.match(p.name) and "pilot" not in p.name:
            rows.append(tabular_row(p))
    for p in sorted(DEEP.glob("sb3_*_summary.json")):
        if SB3_PAT.match(p.name) and "pilot" not in p.name:
            rows.append(deep_row(p))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["return_gap"] = df["optimal_return"] - df["final_return"]
    df["algo"] = pd.Categorical(df["algo"], categories=ALGO_ORDER, ordered=True)
    return df.sort_values(["board", "algo", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------- summary
def _ms(v: pd.Series, digits=1) -> str:
    v = v.dropna().astype(float)
    if len(v) == 0:
        return "-"
    return f"{v.mean():.{digits}f}" if len(v) == 1 else f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def _med(v: pd.Series) -> str:
    v = v.dropna().astype(float)
    return "-" if len(v) == 0 else f"{v.median():.0f} [{v.min():.0f}-{v.max():.0f}]"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (board, algo), g in df.groupby(["board", "algo"], observed=True):
        rows.append({
            "board": board, "algo": NAMES[str(algo)], "n_seeds": len(g), "BFS": int(g["bfs"].iloc[0]),
            "final_moves": _ms(g["final_moves"], 2), "final_success_%": f"{100 * g['final_success'].mean():.0f}",
            "final_return": _ms(g["final_return"], 1), "optimal_return": f"{g['optimal_return'].iloc[0]:.0f}",
            "train_last100_moves": _ms(g["train_last100_moves"], 2),
            "train_last100_return": _ms(g["train_last100_return"], 1),
            "conv_episodes median[min-max]": _med(g["conv_episode"]),
            "conv_episodes mean±sd": _ms(g["conv_episode"], 0),
            "not_converged": int(g["conv_episode"].isna().sum()),
            "conv_timesteps median": "-" if g["conv_timesteps"].notna().sum() == 0 else f"{g['conv_timesteps'].median():.0f}",
            "soft_last20": f"{g['soft_last20'].mean():.2f}",
            "state_agreement": f"{g['state_agreement'].mean():.2f}",
        })
    return pd.DataFrame(rows)


def random_baseline() -> pd.DataFrame:
    rows = []
    for name, b in BOARDS.items():
        succ, mv, ret = run_random_policy(name, episodes=300, seed=0)
        rows.append({"board": name, "BFS": bfs_min_rolls(b), "random_success_%": round(succ, 1),
                     "random_mean_rolls": round(mv, 2), "random_mean_return": round(ret, 1),
                     "optimal_return": optimal_return(b)})
    return pd.DataFrame(rows)


def main():
    df = build_per_run()
    if df.empty:
        print("No results found in results/. Run tabular.py and sb3_train.py first.")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "final_table_per_run.csv", index=False)
    summ = summarize(df)
    summ.to_csv(TABLES / "final_table_summary.csv", index=False)
    rb = random_baseline()
    rb.to_csv(TABLES / "random_baseline.csv", index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print("RANDOM-POLICY BASELINE (300 games, fixed seed) and the best possible return:")
    print(rb.to_string(index=False))
    for board in BOARDS:
        blk = summ[summ["board"] == board]
        if len(blk):
            print(f"\n=== {board} ===")
            print(blk.drop(columns=["board"]).to_string(index=False))
    print("\nn_seeds differs per algorithm; mean ± sd are shown only when n_seeds > 1. "
          "conv_episodes are TRAINING EPISODES. Saved final_table_per_run.csv, final_table_summary.csv, random_baseline.csv")


if __name__ == "__main__":
    main()
