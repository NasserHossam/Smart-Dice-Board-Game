"""
challenge1_random_dice.py - CHALLENGE 1: the environment rolls the dice, the agent does not choose.

HONEST MDP REVISION
  * State  : current square (same as before).
  * Action : NONE. The roll (uniform 1..6) is drawn by the environment. The Gym env has a dummy
             Discrete(1) action only because Gymnasium needs an action space.
  * So the process is a Markov REWARD process (a Markov chain with rewards), not a decision problem:
    there is only one possible policy. Consequences:
      - Win probability and expected game length are properties of the BOARD. They cannot improve
        with training, so training curves for them are flat. We show that, we do not hide it.
      - Q-Learning and SARSA both reduce to the same TD(0) update on V(s) = Q(s, only action),
        because max over one action = the action taken. They give identical tables for the same seed.
      - What RL can learn is the VALUE of each square (expected discounted return). We check it
        against the EXACT value from solving the Markov chain.
  * Rules are unchanged: ladder/snake applied immediately, rolling past N = invalid (stay, -10, the
    roll still counts), rewards 100 / 20 / -1 / -20 / -10, episode cut after 5*N rolls (a failure).

WHAT THE SCRIPT DOES
  1. Self-test: transitions equal the original SmartDiceEnv for every square and roll; a tiny board
     with a hand-checked answer (N=2: expected length 6); Gymnasium check_env.
  2. EXACT analysis per board (linear algebra on the Markov chain): P(win within 5N rolls), P(win within
     10/20/30/50 rolls) (the 5N number is ~1 and uninformative),
     expected game length, expected return, expected snakes/ladders per game, value of every square.
  3. MONTE CARLO simulation of games in the env, compared with the exact numbers.
  4. TD learning (Q-Learning and SARSA, alpha constant, no exploration is needed) over many episodes;
     reports how close the learned values get to the exact values, and whether the two agree.
  5. Tables (CSV) and three PNG plots in results_challenges/ch1/.

Usage (small pilot first to measure time):
  python challenges/challenge1_random_dice.py --episodes 500 --seeds 0 --mc-games 500
Full run (settings are printed and saved):
  python challenges/challenge1_random_dice.py --episodes 20000 --seeds 0 1 2 3 4 --mc-games 20000 --time-limit 120
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # project modules live in src/

from paths import challenge_dir
from env import BOARDS, DEFAULT_REWARDS, Board, SmartDiceEnv, bfs_min_rolls, validate_board

OUT = challenge_dir(1)   # results_challenges/ch1/


# --------------------------------------------------------------------------- transition (one place)
def transition(board: Board, rewards: dict, square: int, roll: int):
    """(new_square, reward, event). Same rules as env.SmartDiceEnv (checked in the self-test)."""
    target = square + roll
    if target > board.n:
        return square, rewards["invalid"], "invalid"          # stay, penalised, roll still counts
    final = board.jump_map.get(target, target)
    if final == board.n:
        return final, rewards["goal"], "goal"
    if target in board.ladders:
        return final, rewards["ladder"], "ladder"
    if target in board.snakes:
        return final, rewards["snake"], "snake"
    return final, rewards["normal"], "normal"


# --------------------------------------------------------------------------- environment
class RandomDiceEnv(gym.Env):
    """Smart Dice board where the ENVIRONMENT rolls the dice (uniform 1..6). Dummy action only."""
    metadata = {"render_modes": []}

    def __init__(self, board="N30", rewards=None, max_steps=None):
        super().__init__()
        self.board = BOARDS[board] if isinstance(board, str) else board
        validate_board(self.board)
        self.n = self.board.n
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.max_steps = int(max_steps) if max_steps is not None else 5 * self.n
        self.observation_space = spaces.Discrete(self.n)     # obs = square - 1
        self.action_space = spaces.Discrete(1)               # dummy: the agent makes no decision
        self.square, self.steps = 1, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.square, self.steps = 1, 0
        return 0, {"square": 1}

    def step(self, action):
        roll = int(self.np_random.integers(1, 7))
        self.steps += 1
        self.square, reward, event = transition(self.board, self.rewards, self.square, roll)
        terminated = self.square == self.n
        truncated = (not terminated) and self.steps >= self.max_steps
        return self.square - 1, float(reward), terminated, truncated, \
            {"square": self.square, "roll": roll, "event": event}


# --------------------------------------------------------------------------- exact analysis
HORIZONS = (10, 20, 30, 50)   # "win within k rolls" is the informative win probability (the 5N cap is almost never hit)


def exact_analysis(board: Board, rewards: dict | None = None, gamma: float = 0.99, cap: int | None = None,
                   horizons=HORIZONS):
    """Exact numbers from the Markov chain over the squares reachable from square 1."""
    r = {**DEFAULT_REWARDS, **(rewards or {})}
    n = board.n
    cap = int(cap) if cap is not None else 5 * n
    seen, stack = {1}, [1]
    while stack:                                              # squares reachable with random dice
        s = stack.pop()
        for roll in range(1, 7):
            ns, _, _ = transition(board, r, s, roll)
            if ns != n and ns not in seen:
                seen.add(ns)
                stack.append(ns)
    T = sorted(seen)
    idx = {s: i for i, s in enumerate(T)}
    m = len(T)
    Qm, g = np.zeros((m, m)), np.zeros(m)
    rbar, p_snake, p_ladder = np.zeros(m), np.zeros(m), np.zeros(m)
    for s in T:
        i = idx[s]
        for roll in range(1, 7):
            ns, rew, ev = transition(board, r, s, roll)
            rbar[i] += rew / 6
            if ev == "goal":
                g[i] += 1 / 6
            else:
                Qm[i, idx[ns]] += 1 / 6
            if ev == "snake":
                p_snake[i] += 1 / 6
            if ev == "ladder":
                p_ladder[i] += 1 / 6
    I = np.eye(m)
    N = np.linalg.inv(I - Qm)                                 # expected visits matrix
    exp_len = float((N @ np.ones(m))[idx[1]])
    exp_ret = float((N @ rbar)[idx[1]])
    v_disc = np.linalg.solve(I - gamma * Qm, rbar)            # discounted value, terminal value 0
    visits = N[idx[1]]
    p, absorbed, p_le = np.zeros(m), 0.0, {}
    p[idx[1]] = 1.0
    for t in range(1, max(cap, max(horizons)) + 1):           # P(reach the goal within t rolls)
        absorbed += float(p @ g)
        p = p @ Qm
        if t in horizons:
            p_le[t] = absorbed
        if t == cap:
            p_cap = absorbed
    return {"states": T, "idx": idx, "v_disc": v_disc, "exp_len": exp_len, "exp_return": exp_ret,
            "p_win_cap": p_cap, "p_win_le": p_le, "exp_snakes": float(visits @ p_snake),
            "exp_ladders": float(visits @ p_ladder), "cap": cap}


# --------------------------------------------------------------------------- Monte Carlo
def monte_carlo(board_name: str, games: int, seed: int, horizons=HORIZONS):
    env = RandomDiceEnv(board_name)
    lens, wins, rets, snk, lad = [], [], [], [], []
    for gm in range(games):
        env.reset(seed=seed if gm == 0 else None)
        done, ret, term, ns, nl = False, 0.0, False, 0, 0
        while not done:
            _, r, term, trunc, info = env.step(0)
            ret += r
            ns += info["event"] == "snake"
            nl += info["event"] == "ladder"
            done = term or trunc
        lens.append(env.steps); wins.append(int(term)); rets.append(ret); snk.append(ns); lad.append(nl)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    lens_a, wins_a = np.array(lens), np.array(wins)
    le = {f"mc_p_win_le_{k}": float(np.mean((wins_a == 1) & (lens_a <= k))) for k in horizons}
    return {**le, "mc_win": float(np.mean(wins)), "mc_win_se": se(wins), "mc_len": float(np.mean(lens)),
            "mc_len_se": se(lens), "mc_return": float(np.mean(rets)), "mc_return_se": se(rets),
            "mc_snakes": float(np.mean(snk)), "mc_ladders": float(np.mean(lad))}


# --------------------------------------------------------------------------- TD learning
def td_learn(board_name: str, algo: str, episodes: int, seed: int, ex: dict,
             alpha: float = 0.05, gamma: float = 0.99, eval_every: int = 250, time_limit=None):
    """Q-Learning / SARSA with ONE action. Returns (Q, log rows, summary)."""
    env = RandomDiceEnv(board_name)
    Q = np.zeros((env.n, 1))
    T = np.array(ex["states"]) - 1
    v_true = ex["v_disc"]
    lens, rets, succ, rows = [], [], [], []
    t0, stopped = time.time(), False
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a, ret, done, term = 0, 0.0, False, False
        while not done:
            nobs, r, term, trunc, _ = env.step(a)
            na = 0                                            # the only action
            if term:
                target = r
            elif algo == "sarsa":
                target = r + gamma * Q[nobs, na]
            else:
                target = r + gamma * Q[nobs].max()
            Q[obs, a] += alpha * (target - Q[obs, a])
            obs, a, ret, done = nobs, na, ret + r, term or trunc
        lens.append(env.steps); rets.append(ret); succ.append(int(term))
        out_of_time = time_limit is not None and time.time() - t0 > time_limit
        if ep % eval_every == 0 or ep == episodes or out_of_time:
            err = np.abs(Q[T, 0] - v_true)
            k = min(eval_every, len(lens))
            rows.append({"board": board_name, "algo": algo, "seed": seed, "episode": ep,
                         "value_mae": float(err.mean()), "start_error": float(Q[0, 0] - v_true[ex["idx"][1]]),
                         "train_len": float(np.mean(lens[-k:])), "train_success": float(np.mean(succ[-k:])),
                         "train_return": float(np.mean(rets[-k:]))})
        if out_of_time:
            stopped = True
            break
    last = rows[-1]
    return Q, rows, {"board": board_name, "algo": algo, "seed": seed, "episodes_run": len(lens),
                     "stopped_by_time_limit": stopped, "elapsed_s": round(time.time() - t0, 2),
                     "final_value_mae": last["value_mae"], "final_start_error": last["start_error"],
                     "exact_start_value": float(v_true[ex["idx"][1]]), "learned_start_value": float(Q[0, 0])}


# --------------------------------------------------------------------------- self-test
def self_test():
    for name, b in BOARDS.items():
        base = SmartDiceEnv(b)
        for s in range(1, b.n):
            for roll in range(1, 7):
                base.reset()
                base.square = s
                obs, r, _, _, info = base.step(roll - 1)
                ns, rew, ev = transition(b, DEFAULT_REWARDS, s, roll)
                assert (obs, r, info["event"]) == (ns - 1, rew, ev), (name, s, roll)
    print("[ok] transition() equals the original SmartDiceEnv for every square and roll, 3 boards")
    tiny = exact_analysis(Board("tiny", 2, {}, {}))
    assert abs(tiny["exp_len"] - 6.0) < 1e-9 and abs(tiny["exp_return"] - 50.0) < 1e-9
    print("[ok] hand-checked tiny board N=2: expected length 6.0, expected return 50.0")
    from gymnasium.utils.env_checker import check_env
    check_env(RandomDiceEnv("N30"), skip_render_check=True)
    print("[ok] gymnasium check_env passed on RandomDiceEnv")


# --------------------------------------------------------------------------- plots
def make_plots(log: pd.DataFrame, ex_all: dict, Qs: dict):
    colors = {"N30": "#1f77b4", "N50": "#ff7f0e", "N100": "#2ca02c"}
    boards = [b for b in BOARDS if b in set(log.board)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in boards:
        g = log[(log.board == b) & (log.algo == "qlearning")].groupby("episode").value_mae
        ax.plot(g.mean().index, g.mean().values, color=colors[b], label=f"{b} (Q-Learning = SARSA)")
        if g.count().max() > 1:
            ax.fill_between(g.mean().index, (g.mean() - g.std()).clip(lower=1e-6), g.mean() + g.std(),
                            color=colors[b], alpha=0.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training episodes"); ax.set_ylabel("mean |learned value - exact value|")
    ax.set_title("Challenge 1: TD learning recovers the exact square values"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ch1_value_error.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(boards), figsize=(5 * len(boards), 4), squeeze=False)
    for ax, b in zip(axes[0], boards):
        for algo, ls in (("qlearning", "-"), ("sarsa", "--")):
            g = log[(log.board == b) & (log.algo == algo)].groupby("episode").train_len.mean()
            ax.plot(g.index, g.values, ls, label=algo)
        ax.axhline(ex_all[b]["exp_len"], color="k", lw=1, label="exact expected length")
        ax.set_title(f"{b}: game length does not improve (no decisions)")
        ax.set_xlabel("training episodes"); ax.set_ylabel("rolls per game (window mean)")
        ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ch1_length_flat.png", dpi=130); plt.close(fig)

    key = next((k for k in Qs if k[0] == "N30"), None)
    if key:
        ex = ex_all["N30"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        x, y = ex["v_disc"], Qs[key][np.array(ex["states"]) - 1, 0]
        ax.scatter(x, y, s=18)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("exact value"); ax.set_ylabel("learned value (after training)")
        ax.set_title(f"N30, seed {key[2]}: learned vs exact square values"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "ch1_learned_vs_exact.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Challenge 1: random dice (no agent decisions)")
    p.add_argument("--boards", nargs="+", default=list(BOARDS), choices=list(BOARDS))
    p.add_argument("--episodes", type=int, required=True, help="TD training episodes per run")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--mc-games", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER TD run; saves what it has")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    self_test()

    ex_all, rows = {}, []
    print("\nEXACT (Markov chain) vs MONTE CARLO (env simulation), random dice, original rewards")
    for b in a.boards:
        t0 = time.time()
        ex = exact_analysis(BOARDS[b], gamma=a.gamma)
        mc = monte_carlo(b, a.mc_games, seed=12345)
        ex_all[b] = ex
        bfs = bfs_min_rolls(BOARDS[b])
        rows.append({"board": b, "bfs_min_rolls_chosen_dice": bfs,
                     "exact_p_win_within_cap": ex["p_win_cap"],
                     **{f"exact_p_win_le_{k}": v for k, v in ex["p_win_le"].items()},
                     **{k: v for k, v in mc.items() if k.startswith("mc_p_win_le_")}, "mc_p_win": mc["mc_win"], "mc_p_win_se": mc["mc_win_se"],
                     "exact_expected_rolls": ex["exp_len"], "mc_expected_rolls": mc["mc_len"], "mc_rolls_se": mc["mc_len_se"],
                     "expected_rolls_over_bfs": ex["exp_len"] / bfs,
                     "exact_expected_return": ex["exp_return"], "mc_return": mc["mc_return"], "mc_return_se": mc["mc_return_se"],
                     "exact_snakes_per_game": ex["exp_snakes"], "mc_snakes_per_game": mc["mc_snakes"],
                     "exact_ladders_per_game": ex["exp_ladders"], "mc_ladders_per_game": mc["mc_ladders"],
                     "cap_rolls": ex["cap"], "mc_games": a.mc_games})
        pd.DataFrame({"square": ex["states"], "exact_discounted_value": ex["v_disc"]}).to_csv(
            OUT / f"ch1_exact_values_{b}.csv", index=False)
        print(f"  {b}: done in {time.time() - t0:.1f}s")
    ex_df = pd.DataFrame(rows)
    ex_df.to_csv(OUT / "ch1_exact_vs_mc.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(ex_df.drop(columns=["exact_snakes_per_game", "mc_snakes_per_game", "exact_ladders_per_game",
                              "mc_ladders_per_game", "cap_rolls", "mc_games"]).round(4).to_string(index=False))
    print("\nSnakes / ladders hit per game and win probability within k rolls (exact vs Monte Carlo):")
    print(ex_df[["board", "exact_snakes_per_game", "mc_snakes_per_game", "exact_ladders_per_game", "mc_ladders_per_game"]
                ].round(3).to_string(index=False))
    print(ex_df[["board"] + [c for c in ex_df.columns if "p_win_le" in c]].round(3).to_string(index=False))

    log_rows, runs, Qs = [], [], {}
    print("\nTD LEARNING (Q-Learning and SARSA have ONE action, so they must agree)")
    for b in a.boards:
        for seed in a.seeds:
            res = {}
            for algo in ("qlearning", "sarsa"):
                Q, lr, summ = td_learn(b, algo, a.episodes, seed, ex_all[b], alpha=a.alpha, gamma=a.gamma,
                                       eval_every=a.eval_every, time_limit=a.time_limit)
                res[algo] = Q
                log_rows += lr; runs.append(summ); Qs[(b, algo, seed)] = Q
                print(f"  {b} {algo:<9} seed {seed}: {summ['episodes_run']} episodes in {summ['elapsed_s']}s, "
                      f"final value MAE {summ['final_value_mae']:.3f}, start value learned "
                      f"{summ['learned_start_value']:.2f} vs exact {summ['exact_start_value']:.2f}"
                      + ("  (STOPPED BY TIME LIMIT)" if summ["stopped_by_time_limit"] else ""), flush=True)
            same = np.array_equal(res["qlearning"], res["sarsa"])
            print(f"      Q-Learning and SARSA tables identical: {same}")
    log, runs_df = pd.DataFrame(log_rows), pd.DataFrame(runs)
    log.to_csv(OUT / "ch1_td_log.csv", index=False)
    runs_df.to_csv(OUT / "ch1_td_runs.csv", index=False)
    print("\nTD summary per board (mean over seeds):")
    print(runs_df.groupby(["board", "algo"])[["final_value_mae", "final_start_error", "exact_start_value",
                                              "learned_start_value", "elapsed_s"]].mean().round(3).to_string())
    make_plots(log, ex_all, Qs)
    print(f"\nSaved tables and ch1_*.png in {OUT}")


if __name__ == "__main__":
    main()
