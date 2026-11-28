"""
challenge6_reward_shaping.py - CHALLENGE 6: reward optimization / reward shaping.

Question from the handout: does reward shaping make learning faster, or does it create unintended behaviour?
Part 1 of Challenge 6 is the reward-function experiment already in experiments.py (five reward functions on
N50, results/exp_reward_N50_*). This script adds Part 2: SHAPING.

DESIGN (choices of mine; the report says so)
  Five reward setups. The agent learns from the setup's reward; every evaluation (rolls, success, return,
  snakes, ladders) uses the ORIGINAL rewards, so all setups are judged on the same task.
    default        original rewards (goal +100, ladder +20, normal -1, snake -20, invalid -10)
    sparse         goal +100, everything else 0 (no hints at all; only the discount favours short games)
    sparse+PBRS    sparse + potential-based shaping   F = gamma * Phi(s') - Phi(s)
    default+PBRS   default + the same potential-based shaping
    progress_bonus default + 2 * (squares moved FORWARD)  (max(0, s' - s); NOT potential-based)
  Potential: Phi(s) = 50 * (s - 1) / (N - 1), i.e. progress along the board (0 on square 1, 50 on the goal).
  It needs no model of the board (only the square number). Potential-based shaping provably keeps the
  optimal policy unchanged (Ng, Harada & Russell, 1999); the progress bonus does not, because a ladder
  followed by a snake pays the forward bonus but never charges the backward move, so a ladder-snake cycle
  can become profitable (reward hacking).
  Algorithms: tabular Q-Learning and SARSA, the same update, alpha 0.1, gamma 0.99, epsilon 1.0 -> 0.05
  over the first 70% of 2,000 episodes, Q starts at 0 (exactly as tabular.py). Boards N50, N100 (main
  project) and N200 (Challenge 5). Greedy evaluation every 10 episodes (every episode for the first 50),
  20 games (only tie-breaking differs). Convergence = main-project rule (success >= 95%, rolls <= 1.10 x BFS,
  from that checkpoint until the end of training).

Usage (pilot first):
  python challenge6_reward_shaping.py --boards N50 --seeds 0 1 --episodes 500
  python challenge6_reward_shaping.py --workers 4
Outputs: results_challenges/ch6_runs.csv, ch6_summary.csv, ch6_curves.csv, ch6_*.png
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import challenge5_dqn_vs_tabular  # noqa: F401  (registers the N200 board in env.BOARDS)
from analyze_tabular import stable_to_end
from env import BOARDS, SmartDiceEnv, bfs_min_rolls
from tabular import epsilon_greedy, greedy_action

OUT = Path(__file__).parent / "results_challenges"
SETUPS = ["default", "sparse", "sparse+PBRS", "default+PBRS", "progress_bonus"]
SPARSE = {"goal": 100.0, "ladder": 0.0, "normal": 0.0, "snake": 0.0, "invalid": 0.0}
PHI_SCALE, BONUS = 50.0, 2.0
ALGOS = {"qlearning": "Q-Learning", "sarsa": "SARSA"}


def phi(square: int, n: int) -> float:
    return PHI_SCALE * (square - 1) / (n - 1)


def learning_reward(setup: str, r_orig: float, s: int, s2: int, n: int, gamma: float) -> float:
    """Reward the agent LEARNS from. s, s2 = square before / after the step; r_orig = original reward."""
    if setup == "default":
        return r_orig
    if setup == "sparse":
        return SPARSE["goal"] if s2 == n else 0.0
    if setup == "sparse+PBRS":
        base = SPARSE["goal"] if s2 == n else 0.0
        return base + gamma * phi(s2, n) - phi(s, n)
    if setup == "default+PBRS":
        return r_orig + gamma * phi(s2, n) - phi(s, n)
    if setup == "progress_bonus":
        return r_orig + BONUS * max(0, s2 - s)
    raise ValueError(setup)


def play_greedy(env: SmartDiceEnv, Q: np.ndarray, rng, games: int):
    """Greedy games under the ORIGINAL rewards. Returns mean success, rolls, return, snakes, ladders."""
    out = []
    for _ in range(games):
        obs, _ = env.reset()
        done, ret, term, sn, la = False, 0.0, False, 0, 0
        while not done:
            obs, r, term, trunc, info = env.step(greedy_action(Q[obs], rng))
            ret += r
            sn += info["event"] == "snake"
            la += info["event"] == "ladder"
            done = term or trunc
        out.append((float(term), env.steps, ret, sn, la))
    return np.mean(np.array(out), axis=0)


def train_shaped(board: str, algo: str, setup: str, seed: int, episodes: int = 2000, alpha: float = 0.1,
                 gamma: float = 0.99, eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_frac: float = 0.7,
                 eval_every: int = 10, dense_until: int = 50, eval_games: int = 20):
    rng = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 10_000)
    env, env_eval = SmartDiceEnv(board), SmartDiceEnv(board)
    n = env.n
    Q = np.zeros((n, 6))
    decay = (eps_end / eps_start) ** (1.0 / max(1, int(eps_decay_frac * episodes)))
    eps = eps_start
    train_moves, evals = [], []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a = epsilon_greedy(Q[obs], eps, rng)
        done = False
        while not done:
            s = obs + 1
            nobs, r, term, trunc, _ = env.step(a)
            rl = learning_reward(setup, r, s, nobs + 1, n, gamma)
            na = epsilon_greedy(Q[nobs], eps, rng)
            if term:
                target = rl
            elif algo == "sarsa":
                target = rl + gamma * Q[nobs, na]
            else:
                target = rl + gamma * Q[nobs].max()
            Q[obs, a] += alpha * (target - Q[obs, a])
            obs, a = nobs, na
            done = term or trunc
        train_moves.append(env.steps)
        eps = max(eps_end, eps * decay)
        if ep <= dense_until or ep % eval_every == 0 or ep == episodes:
            evals.append((ep, *play_greedy(env_eval, Q, rng_eval, eval_games)))
    ev = pd.DataFrame(evals, columns=["episode", "success", "rolls", "return", "snakes", "ladders"])
    return Q, np.array(train_moves), ev


def run_one(board: str, algo: str, setup: str, seed: int, episodes: int, eval_games: int):
    t0 = time.time()
    _, moves, ev = train_shaped(board, algo, setup, seed, episodes=episodes, eval_games=eval_games)
    bfs = bfs_min_rolls(BOARDS[board])
    ok = ((ev.success >= 0.95) & (ev.rolls <= 1.10 * bfs)).to_numpy()
    conv = stable_to_end(ev.episode.to_numpy(), ok)
    early = ev[ev.episode <= 200]
    last = ev.iloc[-1]
    row = {"board": board, "algo": ALGOS[algo], "setup": setup, "seed": seed, "bfs": bfs,
           "final_success": last.success, "final_rolls": last.rolls, "final_return": last["return"],
           "final_snakes": last.snakes, "final_ladders": last.ladders,
           "conv_episode": conv, "conv_steps": None if conv is None else int(moves[:conv].sum()),
           "early_rolls_ratio": float((early.rolls / bfs).mean()),   # mean greedy rolls/BFS, episodes 1-200
           "train_moves_first100": float(moves[:100].mean()), "wall_s": round(time.time() - t0, 1)}
    curve = ev[["episode", "rolls", "success"]].assign(board=board, algo=ALGOS[algo], setup=setup, seed=seed)
    return row, curve


def optimal_under_setup(board: str, setup: str, gamma: float = 0.99):
    """Exact optimal policy for the reward the agent LEARNS from (value iteration on the known model).
    Returns (rolls to the goal, or None if the optimal policy never reaches the goal within 5*N rolls)."""
    b = BOARDS[board]
    n = b.n
    env = SmartDiceEnv(b)
    trans = {}
    for s in range(1, n):
        o = []
        for a in range(6):
            env.square, env.steps = s, 0
            _, r, _, _, _ = env.step(a)
            s2 = env.square
            o.append((learning_reward(setup, r, s, s2, n, gamma), s2))
        trans[s] = o
    v = np.zeros(n + 1)
    for _ in range(200_000):
        new = v.copy()
        for s in range(1, n):
            new[s] = max(r + gamma * v[f] for r, f in trans[s])
        delta = np.max(np.abs(new - v))
        v = new
        if delta < 1e-9:
            break
    s, rolls = 1, 0
    while s != n and rolls < 5 * n:
        r, s = max(trans[s], key=lambda x: x[0] + gamma * v[x[1]])
        rolls += 1
    return rolls if s == n else None


# --------------------------------------------------------------------------- self-test
def self_test():
    n, g = 50, 0.99
    # potential-based shaping telescopes: along any path the shaping terms sum to a quantity that depends
    # only on the start and end squares (for gamma = 1 exactly; checked here with gamma = 1)
    path = [1, 14, 7, 9, 24, 33, 50]
    tot = sum(learning_reward("sparse+PBRS", 0.0, a, b, n, 1.0) for a, b in zip(path, path[1:])) - 100.0
    assert abs(tot - (phi(50, n) - phi(1, n))) < 1e-9
    # progress bonus pays a ladder but does not charge a snake: a ladder-snake cycle earns a positive bonus
    ladder = learning_reward("progress_bonus", 20.0, 3, 14, n, g) - 20.0
    snake = learning_reward("progress_bonus", -20.0, 19, 7, n, g) + 20.0
    assert ladder == BONUS * 11 and snake == 0.0
    # default setup returns the original reward unchanged
    assert learning_reward("default", -1.0, 5, 6, n, g) == -1.0
    print("[ok] shaping terms: PBRS telescopes, progress bonus ignores backward moves, default unchanged")
    # PBRS must not change the optimal policy's length on the main board
    assert optimal_under_setup("N50", "default+PBRS") == optimal_under_setup("N50", "default") == 5
    print("[ok] exact optimum under default+PBRS = default = BFS (5 rolls) on N50")


# --------------------------------------------------------------------------- summary + plots
def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (board, algo, setup), g in runs.groupby(["board", "algo", "setup"], sort=False):
        conv = g.conv_episode.dropna()
        out.append({"board": board, "algo": algo, "setup": setup, "seeds": len(g),
                    "success_%": round(100 * float((g.final_success >= 0.95).mean()), 1),
                    "final_rolls_mean": round(float(g.final_rolls.mean()), 2),
                    "final_return_mean": round(float(g.final_return.mean()), 1),
                    "snakes_mean": round(float(g.final_snakes.mean()), 1),
                    "ladders_mean": round(float(g.final_ladders.mean()), 1),
                    "conv_episodes_median": None if conv.empty else float(np.median(conv)),
                    "conv_steps_median": None if conv.empty else float(np.median(g.conv_steps.dropna())),
                    "not_converged": int(g.conv_episode.isna().sum()),
                    "early_rolls_ratio_median": round(float(g.early_rolls_ratio.median()), 2),
                    "train_moves_first100_mean": round(float(g.train_moves_first100.mean()), 1)})
    s = pd.DataFrame(out)
    s["_n"] = s.board.map(lambda b: BOARDS[b].n)
    s["_k"] = s.setup.map(SETUPS.index)
    return s.sort_values(["_n", "algo", "_k"]).drop(columns=["_n", "_k"])


def make_plots(curves: pd.DataFrame, boards):
    colors = dict(zip(SETUPS, ["black", "tab:orange", "tab:green", "tab:blue", "tab:red"]))
    fig, axes = plt.subplots(2, len(boards), figsize=(5 * len(boards), 8), squeeze=False)
    for j, board in enumerate(boards):
        bfs = bfs_min_rolls(BOARDS[board])
        for i, algo in enumerate(ALGOS.values()):
            a = axes[i, j]
            for setup in SETUPS:
                g = curves[(curves.board == board) & (curves.algo == algo) & (curves.setup == setup)]
                if g.empty:
                    continue
                med = g.groupby("episode").rolls.median()
                a.plot(med.index, med.values / bfs, color=colors[setup], label=setup, lw=1.6)
            a.axhline(1.0, ls="--", color="gray", lw=1)
            a.set_xscale("log"); a.set_ylim(0, 8); a.grid(alpha=0.3)
            a.set_title(f"{board} {algo}: greedy rolls / BFS (median)")
            a.set_xlabel("training episode (log)")
            if j == 0:
                a.set_ylabel("greedy rolls / BFS minimum")
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ch6_shaping_learning_curves.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Challenge 6: reward shaping")
    p.add_argument("--boards", nargs="+", default=["N50", "N100", "N200"])
    p.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    p.add_argument("--setups", nargs="+", default=SETUPS, choices=SETUPS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    self_test()

    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [ex.submit(run_one, b, al, st, s, a.episodes, a.eval_games)
                for b in a.boards for al in a.algos for st in a.setups for s in a.seeds]
        for k, f in enumerate(as_completed(jobs), 1):
            r, c = f.result()
            rows.append(r)
            curves.append(c)
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} runs done", flush=True)
    runs = pd.DataFrame(rows).sort_values(["board", "algo", "setup", "seed"])
    runs.to_csv(OUT / "ch6_runs.csv", index=False)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / "ch6_curves.csv", index=False)
    summary = summarize(runs)
    exact = {(b, st): optimal_under_setup(b, st) for b in a.boards for st in a.setups}
    summary.insert(4, "exact_opt_rolls_for_this_reward",
                   [exact[(b, st)] if exact[(b, st)] is not None else "never finishes"
                    for b, st in zip(summary.board, summary.setup)])
    summary.to_csv(OUT / "ch6_summary.csv", index=False)
    make_plots(curves, a.boards)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print("\nSaved: ch6_runs.csv, ch6_curves.csv, ch6_summary.csv, ch6_shaping_learning_curves.png")


if __name__ == "__main__":
    main()
