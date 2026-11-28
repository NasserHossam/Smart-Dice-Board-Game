"""
challenge3_dynamic_board.py - CHALLENGE 3: the snakes and ladders change between episodes.

DESIGN (all of it is a choice of mine; the report should say so)
  * The agent still CHOOSES the dice (action a = roll a+1), exactly as in the original game.
  * Every episode starts on a layout drawn at random from a POOL of layouts. All layouts have N squares
    (default 50), 5 ladders and 5 snakes, jump length between 8 and 16, and obey the original rules
    (no jump on square 1 or N, one jump per square, no chaining, goal reachable).
    The original fixed N50 board satisfies all of these ranges, so it is a typical member of this family.
  * Two observation designs are compared (the agent never sees the layout directly):
      pos   : Discrete(N), the square only. The agent cannot tell which layout it is playing, so it can
              only learn one average policy per square.
      local : Discrete(5^6), what each of the next 6 rolls would do from here:
              0 invalid (overshoots N), 1 snake, 2 normal, 3 ladder, 4 reaches the goal.
              This is a "local view": it does not contain the square number or the layout.
  * GENERALIZATION: train on a pool of M layouts (M = 1, 5, 25, 200), then test on 200 HELD-OUT layouts
    (different layout ids, checked to be different boards) and also on layouts seen in training.
    Metric: excess rolls = rolls used - BFS minimum for that layout (0 is perfect), over games that
    reached the goal, plus success rate. The environment is deterministic given the layout, so each test
    layout is played once with the greedy policy (random tie-breaking).
  * ADAPTATION: the position-only agent trained on ONE layout (M = 1) is moved to a new layout and keeps
    learning there (epsilon 0.1). We record how well it plays immediately and how many episodes it needs
    to recover. The local-view agent trained on M = 200 layouts is tested on the same new layouts WITHOUT
    any extra training.
  * Baselines on the same held-out layouts: uniformly random rolls, and a hand-written rule that uses
    only the local view (no learning). The BFS optimum has excess 0 by definition (it knows the layout).
  * Algorithm: Q-Learning by default (use --algos qlearning sarsa to add SARSA). alpha 0.1, gamma 0.99,
    epsilon 1.0 -> 0.05 over the first 70% of training.

Usage (pilot first):
  python challenges/challenge3_dynamic_board.py --episodes 3000 --seeds 0 --n-test 30 --adapt-layouts 5
  python challenges/challenge3_dynamic_board.py --episodes 30000 --seeds 0 1 2
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
from env import BOARDS, DEFAULT_REWARDS, Board, SmartDiceEnv, bfs_distances_to_goal, validate_board

OUT = challenge_dir(3)   # results_challenges/ch3/
INVALID, SNAKE, NORMAL, LADDER, GOAL = 0, 1, 2, 3, 4
N_VIEWS = 5 ** 6
TRAIN_BASE, TEST_BASE = 0, 100_000          # layout ids: training pool uses 0.., held-out uses 100000..


# --------------------------------------------------------------------------- layouts
def generate_layout(layout_id: int, n: int = 50, k_ladders: int = 5, k_snakes: int = 5,
                    lmin: int = 8, lmax: int = 16) -> Board:
    """Random valid board, fully determined by layout_id."""
    rng = np.random.default_rng([3030, int(layout_id)])
    for _ in range(1000):
        ladders, snakes, starts, ends, ok = {}, {}, set(), set(), True
        for kind, count in (("L", k_ladders), ("S", k_snakes)):
            for _ in range(count):
                placed = False
                for _try in range(300):
                    length = int(rng.integers(lmin, lmax + 1))
                    s = int(rng.integers(2, n))                       # 2 .. n-1
                    e = s + length if kind == "L" else s - length
                    if not 1 <= e <= n:
                        continue
                    if s in starts or s in ends or e in starts:       # one jump per square, no chaining
                        continue
                    (ladders if kind == "L" else snakes)[s] = e
                    starts.add(s)
                    ends.add(e)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        board = Board(f"dyn{layout_id}", n, ladders, snakes)
        try:
            validate_board(board)
        except ValueError:
            continue
        return board
    raise RuntimeError("could not generate a layout")


class Layout:
    """A board plus lookup tables: next square, reward, outcome category, local-view code, BFS optimum."""

    def __init__(self, board: Board):
        n = board.n
        self.board, self.n = board, n
        self.nxt = np.zeros((n + 1, 7), dtype=int)
        self.rew = np.zeros((n + 1, 7))
        self.cat = np.zeros((n + 1, 7), dtype=int)
        self.view = np.zeros(n + 1, dtype=int)
        R = DEFAULT_REWARDS
        for s in range(1, n):
            for r in range(1, 7):
                target = s + r
                if target > n:
                    self.nxt[s, r], self.rew[s, r], self.cat[s, r] = s, R["invalid"], INVALID
                    continue
                final = board.jump_map.get(target, target)
                self.nxt[s, r] = final
                if final == n:
                    self.rew[s, r], self.cat[s, r] = R["goal"], GOAL
                elif target in board.ladders:
                    self.rew[s, r], self.cat[s, r] = R["ladder"], LADDER
                elif target in board.snakes:
                    self.rew[s, r], self.cat[s, r] = R["snake"], SNAKE
                else:
                    self.rew[s, r], self.cat[s, r] = R["normal"], NORMAL
            self.view[s] = sum(int(self.cat[s, r]) * 5 ** (r - 1) for r in range(1, 7))
        self.bfs_start = int(bfs_distances_to_goal(board)[1])

    def key(self):
        return frozenset(self.board.ladders.items()), frozenset(self.board.snakes.items())


# --------------------------------------------------------------------------- environment
class DynamicBoardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layouts, obs_mode: str = "pos", max_steps: int | None = None):
        super().__init__()
        self.layouts = list(layouts)
        self.n = self.layouts[0].n
        assert all(L.n == self.n for L in self.layouts), "all layouts must have the same N"
        assert obs_mode in ("pos", "local")
        self.obs_mode = obs_mode
        self.max_steps = int(max_steps) if max_steps is not None else 5 * self.n
        self.observation_space = spaces.Discrete(self.n if obs_mode == "pos" else N_VIEWS)
        self.action_space = spaces.Discrete(6)
        self.layout, self.layout_index, self.square, self.steps = self.layouts[0], 0, 1, 0

    def _obs(self) -> int:
        return self.square - 1 if self.obs_mode == "pos" else int(self.layout.view[self.square])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "layout_index" in options:
            idx = int(options["layout_index"])
        else:
            idx = int(self.np_random.integers(len(self.layouts)))
        self.layout, self.layout_index = self.layouts[idx], idx
        self.square, self.steps = 1, 0
        return self._obs(), {"layout_index": idx}

    def step(self, action):
        action = int(action)
        if not 0 <= action <= 5:
            raise ValueError(f"action must be in 0..5, got {action}")
        L, roll = self.layout, action + 1
        self.steps += 1
        reward = float(L.rew[self.square, roll])
        self.square = int(L.nxt[self.square, roll])
        terminated = self.square == self.n
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"square": self.square}


# --------------------------------------------------------------------------- evaluation
def evaluate_policy(layouts, chooser, mode: str = "pos", repeats: int = 1, seed: int = 0) -> dict:
    """Play each layout `repeats` times. chooser(obs, env, rng) -> action."""
    env = DynamicBoardEnv(layouts, mode)
    rng = np.random.default_rng(seed)
    wins, rolls, excess = [], [], []
    for i, L in enumerate(layouts):
        for _ in range(repeats):
            obs, _ = env.reset(options={"layout_index": i})
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(chooser(obs, env, rng))
                done = term or trunc
            wins.append(bool(term))
            rolls.append(env.steps)
            if term:
                excess.append(env.steps - L.bfs_start)
    return {"success": float(np.mean(wins)), "mean_rolls": float(np.mean(rolls)),
            "mean_excess": float(np.mean(excess)) if excess else float("nan"),
            "median_excess": float(np.median(excess)) if excess else float("nan")}


def q_chooser(Q, visits=None, counter=None):
    def f(obs, env, rng):
        q = Q[obs]
        idx = np.flatnonzero(q == q.max())
        if counter is not None and visits is not None:
            counter[0] += 1
            counter[1] += int(visits[obs] == 0)
        return int(idx[0]) if len(idx) == 1 else int(rng.choice(idx))
    return f


def evaluate_q(Q, mode, layouts, visits=None, repeats=1, seed=0) -> dict:
    counter = [0, 0]
    out = evaluate_policy(layouts, q_chooser(Q, visits, counter), mode, repeats, seed)
    out["unseen_state_share"] = counter[1] / counter[0] if counter[0] else float("nan")
    return out


def random_chooser(obs, env, rng):
    return int(rng.integers(0, 6))


def heuristic_chooser(obs, env, rng):
    """Hand-written rule using ONLY the local view (the 6 outcome categories): goal > ladder > normal >
    invalid > snake, ties broken by the larger roll. It does not look at the square or the ladder heights."""
    L, s = env.layout, env.square
    prio = {GOAL: 5, LADDER: 4, NORMAL: 3, INVALID: 2, SNAKE: 1}
    best = max(range(1, 7), key=lambda r: (prio[int(L.cat[s, r])], r))
    return best - 1


# --------------------------------------------------------------------------- TD learning
def train(mode, pool, episodes, seed, algo="qlearning", alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
          decay_frac=0.7, Q=None, visits=None, eval_layouts=None, eval_every=None, eval_repeats=1,
          time_limit=None):
    env = DynamicBoardEnv(pool, mode)
    rng = np.random.default_rng(seed + 777)
    nobs = env.observation_space.n
    Q = np.zeros((nobs, 6)) if Q is None else Q.copy()
    visits = np.zeros(nobs, dtype=int) if visits is None else visits.copy()
    decay_eps = max(1, int(decay_frac * episodes))
    log, t0, ep = [], time.time(), 0

    def pick(o, eps):
        if rng.random() < eps:
            return int(rng.integers(0, 6))
        q = Q[o]
        idx = np.flatnonzero(q == q.max())
        return int(idx[0]) if len(idx) == 1 else int(rng.choice(idx))

    for ep in range(1, episodes + 1):
        eps = max(eps_end, eps_start * (eps_end / eps_start) ** (ep / decay_eps))
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a = pick(obs, eps)
        while True:
            visits[obs] += 1
            obs2, r, term, trunc, _ = env.step(a)
            if term:
                target, a2 = r, 0
            else:
                a2 = pick(obs2, eps)
                nxt = Q[obs2].max() if algo == "qlearning" else Q[obs2, a2]
                target = r + gamma * nxt
            Q[obs, a] += alpha * (target - Q[obs, a])
            if term or trunc:
                break
            obs, a = obs2, a2
        if eval_layouts is not None and eval_every and (ep % eval_every == 0):
            e = evaluate_q(Q, mode, eval_layouts, visits, eval_repeats, seed=seed + ep)
            log.append({"episode": ep, **e})
        if time_limit is not None and time.time() - t0 > time_limit:
            break
    return Q, visits, pd.DataFrame(log), time.time() - t0, ep


def stable_to_end(episodes, good):
    if len(good) == 0 or not good[-1]:
        return None
    k = len(good) - 1
    while k > 0 and good[k - 1]:
        k -= 1
    return int(episodes[k])


# --------------------------------------------------------------------------- self-test
def self_test(args):
    # 1) generator: valid boards, correct counts and jump lengths, deterministic
    for i in range(300):
        b = generate_layout(TRAIN_BASE + i, args.n, args.k, args.k, args.lmin, args.lmax)
        validate_board(b)
        assert len(b.ladders) == args.k and len(b.snakes) == args.k
        assert all(args.lmin <= e - s <= args.lmax for s, e in b.ladders.items())
        assert all(args.lmin <= s - e <= args.lmax for s, e in b.snakes.items())
        assert b.ladders == generate_layout(TRAIN_BASE + i, args.n, args.k, args.k, args.lmin, args.lmax).ladders
    print("[ok] 300 generated layouts are valid, have the right counts and jump lengths, and are reproducible")

    # 2) the dynamic env on a fixed layout equals the original SmartDiceEnv, every square and roll
    for name in ("N30", "N50", "N100"):
        L = Layout(BOARDS[name])
        env = DynamicBoardEnv([L], "pos")
        orig = SmartDiceEnv(name)
        for s in range(1, L.n):
            for r in range(1, 7):
                env.reset()
                env.square = s
                o1, r1, t1, _, _ = env.step(r - 1)
                orig.reset()
                orig.square = s
                o2, r2, t2, _, _ = orig.step(r - 1)
                assert (o1, r1, t1) == (o2, r2, t2), (name, s, r)
    print("[ok] dynamic env on a fixed board equals the original SmartDiceEnv for every square and roll")

    # 3) hand-checked local view on N30: ladders {5:13, 11:19, 20:26}, snakes {14:4, 23:9, 27:15}
    L = Layout(BOARDS["N30"])
    assert [int(L.cat[1, r]) for r in range(1, 7)] == [NORMAL, NORMAL, NORMAL, LADDER, NORMAL, NORMAL]
    assert int(L.view[1]) == 2 + 2 * 5 + 2 * 25 + 3 * 125 + 2 * 625 + 2 * 3125
    assert [int(L.cat[28, r]) for r in range(1, 7)] == [NORMAL, GOAL, INVALID, INVALID, INVALID, INVALID]
    assert int(L.cat[8, 6]) == SNAKE and int(L.nxt[8, 6]) == 4
    assert int(L.cat[26, 4]) == GOAL and L.bfs_start == 4
    print("[ok] hand-checked local view on N30 (ladder, normal, goal, invalid, snake) and BFS optimum")

    # 3b) the local view does not contain the square number: two squares with equal view share a code
    assert len({int(L.view[s]) for s in range(1, L.n)}) < L.n - 1
    print("[ok] local view merges squares that look alike (it is not a hidden copy of the position)")

    # 4) the BFS-optimal policy reaches the goal in exactly bfs_start rolls on random layouts
    layouts = [Layout(generate_layout(TEST_BASE + i, args.n, args.k, args.k, args.lmin, args.lmax)) for i in range(40)]
    from env import optimal_roll_sets
    env = DynamicBoardEnv(layouts, "pos")
    for i, Lx in enumerate(layouts):
        sets = optimal_roll_sets(Lx.board)
        obs, _ = env.reset(options={"layout_index": i})
        done = False
        while not done:
            a = min(sets[env.square]) - 1
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        assert term and env.steps == Lx.bfs_start
    print("[ok] BFS-optimal play reaches the goal in exactly the BFS number of rolls on 40 random layouts")

    # 5) train / held-out pools are different boards
    train_keys = {Layout(generate_layout(TRAIN_BASE + i, args.n, args.k, args.k, args.lmin, args.lmax)).key()
                  for i in range(max(args.pools))}
    test_keys = {Layout(generate_layout(TEST_BASE + i, args.n, args.k, args.k, args.lmin, args.lmax)).key()
                 for i in range(args.n_test)}
    assert not (train_keys & test_keys)
    print("[ok] no held-out layout equals a training layout")

    # 6) Gymnasium's checker, both observation designs
    from gymnasium.utils.env_checker import check_env
    for mode in ("pos", "local"):
        check_env(DynamicBoardEnv(layouts[:5], mode), skip_render_check=True)
    print("[ok] gymnasium check_env passed for both observation designs")


# --------------------------------------------------------------------------- plots
def make_plots(gen: pd.DataFrame, curves: pd.DataFrame, adapt_curve: pd.DataFrame, adapt_zero: dict, base: dict):
    # 1) generalization: excess rolls vs number of training layouts
    fig, ax = plt.subplots(figsize=(7, 4.6))
    pools = sorted(gen.pool.unique())
    for mode, color in (("pos", "tab:blue"), ("local", "tab:orange")):
        for split, ls in (("heldout", "-"), ("seen", "--")):
            d = gen[(gen["mode"] == mode) & (gen.split == split)].groupby("pool").mean_excess
            med, lo, hi = d.median(), d.min(), d.max()
            ax.errorbar(pools, [med[p] for p in pools],
                        yerr=[[med[p] - lo[p] for p in pools], [hi[p] - med[p] for p in pools]],
                        color=color, ls=ls, marker="o", capsize=3,
                        label=f"{mode}, {'held-out' if split == 'heldout' else 'seen'} layouts")
    ax.axhline(base["heuristic_mean_excess"], color="gray", ls=":", label="hand-written local rule")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(pools)
    ax.set_xticklabels([str(p) for p in pools])
    ax.set_xlabel("number of training layouts M")
    ax.set_ylabel("mean excess rolls over BFS optimum (median of seeds, bars = min-max)")
    ax.set_title("Generalization to new boards")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ch3_generalization.png", dpi=130)
    plt.close(fig)

    # 2) learning curves on held-out layouts
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mode in zip(axes, ("pos", "local")):
        for pool in pools:
            d = curves[(curves["mode"] == mode) & (curves.pool == pool)]
            piv = d.pivot_table(index="episode", columns="seed", values="mean_excess")
            ax.plot(piv.index, piv.median(axis=1), label=f"M={pool}")
        ax.set_title(f"{mode}: held-out layouts during training")
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("mean excess rolls over BFS (median of seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "ch3_learning_curves.png", dpi=130)
    plt.close(fig)

    # 3) adaptation
    if not adapt_curve.empty:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        piv = adapt_curve.pivot_table(index="episode", columns=["seed", "layout"], values="mean_excess")
        ax.plot(piv.index, piv.median(axis=1), color="tab:blue", label="position-only agent, keeps learning on the new board")
        ax.fill_between(piv.index, piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1), color="tab:blue", alpha=0.2)
        ax.axhline(adapt_zero["local_median"], color="tab:orange", ls="--",
                   label="local-view agent (M=200), no extra training")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("episodes on the new board")
        ax.set_ylabel("mean excess rolls over BFS (median, band = quartiles)")
        ax.set_title("Adaptation to a changed board")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "ch3_adaptation.png", dpi=130)
        plt.close(fig)


# --------------------------------------------------------------------------- main
def med_rng(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return "n/a"
    return f"{x.median():.2f} [{x.min():.2f}-{x.max():.2f}]"


def main():
    p = argparse.ArgumentParser(description="Challenge 3: dynamic board")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--k", type=int, default=5, help="ladders and snakes per layout")
    p.add_argument("--lmin", type=int, default=8)
    p.add_argument("--lmax", type=int, default=16)
    p.add_argument("--pools", type=int, nargs="+", default=[1, 5, 25, 200], help="training pool sizes M (must include 1)")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--algos", nargs="+", default=["qlearning"], choices=["qlearning", "sarsa"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--adapt-layouts", type=int, default=20)
    p.add_argument("--adapt-episodes", type=int, default=600)
    p.add_argument("--adapt-every", type=int, default=20)
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER training run")
    a = p.parse_args()
    assert 1 in a.pools, "--pools must include 1 (needed for the adaptation experiment)"
    a.pools = sorted(a.pools)
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    self_test(a)

    gen_args = dict(n=a.n, k_ladders=a.k, k_snakes=a.k, lmin=a.lmin, lmax=a.lmax)
    train_all = [Layout(generate_layout(TRAIN_BASE + i, **gen_args)) for i in range(max(a.pools))]
    heldout = [Layout(generate_layout(TEST_BASE + i, **gen_args)) for i in range(a.n_test)]
    bfs = [L.bfs_start for L in heldout]
    print(f"\nHeld-out layouts: {len(heldout)} boards, N={a.n}, BFS minimum rolls median {np.median(bfs):.0f} "
          f"[{min(bfs)}-{max(bfs)}]")

    # --- baselines (no learning)
    rnd = evaluate_policy(heldout, random_chooser, "pos", repeats=20, seed=1)
    heu = evaluate_policy(heldout, heuristic_chooser, "pos", repeats=1, seed=1)
    base = {"random_mean_excess": rnd["mean_excess"], "heuristic_mean_excess": heu["mean_excess"]}
    base_df = pd.DataFrame([
        {"policy": "BFS optimum (knows the layout)", "success": 1.0, "mean_excess": 0.0},
        {"policy": "hand-written local rule", "success": heu["success"], "mean_excess": heu["mean_excess"]},
        {"policy": "uniformly random rolls", "success": rnd["success"], "mean_excess": rnd["mean_excess"]}])
    base_df.to_csv(OUT / "ch3_baselines.csv", index=False)
    print("\nBASELINES on the held-out layouts (excess = rolls - BFS minimum, over games that reached the goal)")
    print(base_df.round(3).to_string(index=False))

    # --- experiment 1: generalization
    rows, curve_logs, Qs = [], [], {}
    eval_curve_layouts = heldout[:min(50, len(heldout))]
    for algo in a.algos:
        for mode in ("pos", "local"):
            for pool_size in a.pools:
                pool = train_all[:pool_size]
                for seed in a.seeds:
                    Q, vis, log, secs, ep = train(mode, pool, a.episodes, seed, algo, a.alpha, a.gamma,
                                                  eval_layouts=eval_curve_layouts, eval_every=a.eval_every,
                                                  time_limit=a.time_limit)
                    Qs[(algo, mode, pool_size, seed)] = (Q, vis)
                    if not log.empty:
                        log = log.assign(algo=algo, mode=mode, pool=pool_size, seed=seed)
                        curve_logs.append(log)
                    for split, layouts in (("seen", pool[:min(pool_size, 50)]), ("heldout", heldout)):
                        e = evaluate_q(Q, mode, layouts, vis, seed=seed)
                        rows.append({"algo": algo, "mode": mode, "pool": pool_size, "seed": seed, "split": split,
                                     "episodes_run": ep, "elapsed_s": round(secs, 1), **e})
                    h = rows[-1]
                    s_ = rows[-2]
                    print(f"[{algo} {mode} M={pool_size} seed{seed}] {ep} ep, {secs:.1f}s | held-out: success "
                          f"{h['success']*100:.0f}% excess {h['mean_excess']:.2f} unseen-state steps "
                          f"{h['unseen_state_share']*100:.1f}% | seen: success {s_['success']*100:.0f}% "
                          f"excess {s_['mean_excess']:.2f}", flush=True)
    gen = pd.DataFrame(rows)
    gen.to_csv(OUT / "ch3_generalization_runs.csv", index=False)
    curves = pd.concat(curve_logs, ignore_index=True) if curve_logs else pd.DataFrame()
    curves.to_csv(OUT / "ch3_learning_curves.csv", index=False)

    summ = []
    for (algo, mode, pool_size, split), g in gen.groupby(["algo", "mode", "pool", "split"], sort=False):
        summ.append({"algo": algo, "mode": mode, "M": pool_size, "split": split, "n_seeds": len(g),
                     "success_%": med_rng(g.success * 100), "mean_excess_rolls": med_rng(g.mean_excess),
                     "unseen_state_steps_%": med_rng(g.unseen_state_share * 100)})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "ch3_generalization_summary.csv", index=False)
    print("\nGENERALIZATION (median [min-max] over seeds)")
    print(summ.to_string(index=False))

    # --- experiment 2: adaptation (first algorithm only)
    algo0 = a.algos[0]
    big = max(a.pools)
    adapt_rows, adapt_curves = [], []
    new_boards = heldout[:min(a.adapt_layouts, len(heldout))]
    for seed in a.seeds:
        Qp, vp = Qs[(algo0, "pos", 1, seed)]
        Ql, vl = Qs[(algo0, "local", big, seed)]
        for bi, B in enumerate(new_boards):
            z_pos = evaluate_q(Qp, "pos", [B], vp, repeats=5, seed=seed)
            z_loc = evaluate_q(Ql, "local", [B], vl, repeats=5, seed=seed)
            _, _, log, _, _ = train("pos", [B], a.adapt_episodes, seed * 1000 + bi, algo0, a.alpha, a.gamma,
                                    eps_start=0.1, eps_end=0.1, Q=Qp, visits=vp, eval_layouts=[B],
                                    eval_every=a.adapt_every, eval_repeats=3)
            zero = pd.DataFrame([{"episode": 0, "success": z_pos["success"], "mean_excess": z_pos["mean_excess"]}])
            log = pd.concat([zero, log[["episode", "success", "mean_excess"]]], ignore_index=True)
            good = ((log.success == 1.0) & (log.mean_excess <= 1.0)).to_numpy()
            rec = stable_to_end(log.episode.to_numpy(), good)
            adapt_curves.append(log.assign(seed=seed, layout=bi))
            adapt_rows.append({"seed": seed, "layout": bi, "bfs": B.bfs_start,
                               "pos_zero_success": z_pos["success"], "pos_zero_excess": z_pos["mean_excess"],
                               "local_zero_success": z_loc["success"], "local_zero_excess": z_loc["mean_excess"],
                               "recovery_episode": rec})
    adapt = pd.DataFrame(adapt_rows)
    adapt.to_csv(OUT / "ch3_adaptation_runs.csv", index=False)
    adapt_curve = pd.concat(adapt_curves, ignore_index=True)
    adapt_curve.to_csv(OUT / "ch3_adaptation_curve.csv", index=False)
    rec_ok = adapt.recovery_episode.notna()
    print(f"\nADAPTATION: {len(adapt)} (seed, new board) cases. 'Recovered' = from that episode on, the greedy policy "
          f"reaches the goal within BFS + 1 rolls (checked every {a.adapt_every} episodes, up to {a.adapt_episodes}).")
    print(f"  position-only agent trained on 1 layout, no retraining: success {adapt.pos_zero_success.mean()*100:.0f}%, "
          f"excess rolls median {adapt.pos_zero_excess.median():.2f}")
    print(f"  local-view agent trained on {big} layouts, no retraining: success {adapt.local_zero_success.mean()*100:.0f}%, "
          f"excess rolls median {adapt.local_zero_excess.median():.2f}")
    print(f"  position-only agent after retraining: recovered in {rec_ok.mean()*100:.0f}% of cases; "
          f"episodes to recover (median of those) {adapt.recovery_episode.median():.0f}")
    make_plots(gen[gen.algo == algo0], curves[curves.algo == algo0] if not curves.empty else curves,
               adapt_curve, {"local_median": float(adapt.local_zero_excess.median())}, base)
    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
