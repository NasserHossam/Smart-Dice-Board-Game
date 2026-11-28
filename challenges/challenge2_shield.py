"""
challenge2_shield.py - CHALLENGE 2: limited special action (one snake shield per episode).

HONEST MDP DESIGN
  * Dice: random (as in Challenge 1). With CHOSEN dice the shield would be pointless, because the
    optimal player already hits 0 snakes. With random dice the shield has real value.
  * Extra state: shield_left in {0, 1}. The episode starts with shield_left = 1.
  * Observation the agent sees BEFORE acting: (square, shield_left, rolled value). The roll is drawn
    by the environment, then shown. Observation = ((square-1)*2 + shield_left)*6 + (roll-1), Discrete(12*N).
  * Action: Discrete(2). 0 = accept the roll as is. 1 = activate the shield for this roll.
    The shield only has an effect if (shield_left == 1 and the roll lands on a snake head). In every
    other situation action 1 behaves exactly like action 0 and the shield is NOT spent.
  * Shield effect: the snake is cancelled. The agent stays on the snake-head square, gets the
    normal-square reward (-1) instead of -20, and shield_left becomes 0.
  * Everything else is unchanged: ladders, invalid moves (stay, -10, counts as a roll), rewards
    100 / 20 / -1 / -20 / -10, episode cut after 5*N rolls (counts as a failure).
  * "Decision states" = (square, roll) pairs with shield_left == 1 and square + roll on a snake head.
    These are the ONLY places where the agent's choice changes anything.

WHAT THE SCRIPT DOES
  1. Self-test (transitions vs the original SmartDiceEnv, hand-checked cases on a tiny board, DP
     optimality, Gymnasium check_env, simulation vs exact).
  2. EXACT analysis (linear algebra on the Markov chain over (square, shield)) for three fixed policies:
     never use the shield, use it on the first snake hit, and the OPTIMAL policy found by policy
     iteration on the discounted return. Reports expected rolls, return, snakes hit, P(shield spent).
  3. Monte Carlo simulation in the environment, compared with the exact numbers.
  4. Q-Learning and SARSA on the 12*N-observation environment. Their greedy policy is evaluated EXACTLY
     (no sampling noise) and compared with the optimal policy: regret in discounted return and the
     share of decision states where it agrees with the optimal choice.
  5. CSV tables and PNG plots in results_challenges/ch2/.

Usage (pilot first):
  python challenges/challenge2_shield.py --episodes 3000 --seeds 0 --mc-games 2000
  python challenges/challenge2_shield.py --episodes 50000 --seeds 0 1 2 3 4 --mc-games 10000 --time-limit 300
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
from env import BOARDS, DEFAULT_REWARDS, Board, SmartDiceEnv, validate_board

OUT = challenge_dir(2)   # results_challenges/ch2/
NAMES = {"qlearning": "Q-Learning", "sarsa": "SARSA"}


# --------------------------------------------------------------------------- transition rule
def move(board: Board, rewards: dict, square: int, roll: int, shield: int, use: bool):
    """One roll. Returns (new_square, reward, event, new_shield)."""
    target = square + roll
    if target > board.n:
        return square, rewards["invalid"], "invalid", shield
    final = board.jump_map.get(target, target)
    if final == board.n:
        return final, rewards["goal"], "goal", shield
    if target in board.ladders:
        return final, rewards["ladder"], "ladder", shield
    if target in board.snakes:
        if shield == 1 and use:
            return target, rewards["normal"], "shielded", 0
        return final, rewards["snake"], "snake", shield
    return final, rewards["normal"], "normal", shield


def decision_states(board: Board):
    """All (square, roll) where the shield choice matters."""
    return [(s, r) for s in range(1, board.n) for r in range(1, 7) if s + r in board.snakes]


# --------------------------------------------------------------------------- environment
class ShieldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, board: Board | str = "N30", rewards: dict | None = None, max_steps: int | None = None):
        super().__init__()
        self.board = BOARDS[board] if isinstance(board, str) else board
        validate_board(self.board)
        self.n = self.board.n
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.max_steps = int(max_steps) if max_steps is not None else 5 * self.n
        self.observation_space = spaces.Discrete(self.n * 2 * 6)
        self.action_space = spaces.Discrete(2)
        self.square, self.shield, self.roll, self.steps = 1, 1, 1, 0

    def _obs(self) -> int:
        return int(((self.square - 1) * 2 + self.shield) * 6 + (self.roll - 1))

    def _draw(self):
        self.roll = int(self.np_random.integers(1, 7))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.square, self.shield, self.steps = 1, 1, 0
        self._draw()
        return self._obs(), {"square": 1, "shield": 1, "roll": self.roll}

    def step(self, action):
        action = int(action)
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action}")
        self.steps += 1
        used_roll = self.roll
        self.square, reward, event, self.shield = move(
            self.board, self.rewards, self.square, used_roll, self.shield, action == 1)
        terminated = self.square == self.n
        truncated = (not terminated) and self.steps >= self.max_steps
        if not terminated:
            self._draw()
        info = {"square": self.square, "shield": self.shield, "roll": used_roll, "event": event}
        return self._obs(), float(reward), terminated, truncated, info


def decode(obs: int):
    sq_sh, r = divmod(int(obs), 6)
    sq0, sh = divmod(sq_sh, 2)
    return sq0 + 1, sh, r + 1


# --------------------------------------------------------------------------- exact analysis
def _idx(sq: int, sh: int) -> int:
    return (sq - 1) * 2 + sh


def build_chain(board: Board, rewards: dict, use: np.ndarray):
    """Markov chain over non-terminal (square, shield) states for a fixed policy.
    use[sq, roll] in {0,1} matters only in decision states. Returns P and one-step expected features."""
    n = board.n
    S = 2 * (n - 1)
    P = np.zeros((S, S))
    feat = {k: np.zeros(S) for k in ("reward", "steps", "snakes", "used")}
    for sq in range(1, n):
        for sh in (0, 1):
            i = _idx(sq, sh)
            for r in range(1, 7):
                nsq, rew, ev, nsh = move(board, rewards, sq, r, sh, bool(use[sq, r]))
                if nsq != n:
                    P[i, _idx(nsq, nsh)] += 1 / 6
                feat["reward"][i] += rew / 6
                feat["steps"][i] += 1 / 6
                feat["snakes"][i] += (ev == "snake") / 6
                feat["used"][i] += (ev == "shielded") / 6
    return P, feat


def policy_value(board: Board, rewards: dict, use: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted value V[sq, sh] (shape (n+1, 2), terminal = 0) of the fixed policy."""
    P, feat = build_chain(board, rewards, use)
    v = np.linalg.solve(np.eye(P.shape[0]) - gamma * P, feat["reward"])
    V = np.zeros((board.n + 1, 2))
    for sq in range(1, board.n):
        for sh in (0, 1):
            V[sq, sh] = v[_idx(sq, sh)]
    return V


def policy_metrics(board: Board, rewards: dict, use: np.ndarray, gamma: float = 0.99) -> dict:
    """Expected game statistics from the start (square 1, shield 1), undiscounted, plus discounted value."""
    P, feat = build_chain(board, rewards, use)
    A = np.eye(P.shape[0]) - P
    s = _idx(1, 1)
    out = {k: float(np.linalg.solve(A, v)[s]) for k, v in feat.items()}
    return {"exp_rolls": out["steps"], "exp_return": out["reward"], "exp_snakes": out["snakes"],
            "p_shield_used": out["used"], "disc_value": float(policy_value(board, rewards, use, gamma)[1, 1])}


def optimal_policy(board: Board, rewards: dict, gamma: float = 0.99, max_iter: int = 100):
    """Policy iteration on the discounted return. Returns (use array, V)."""
    n = board.n
    use = np.zeros((n + 1, 7), dtype=int)
    dec = decision_states(board)
    for _ in range(max_iter):
        V = policy_value(board, rewards, use, gamma)
        new = use.copy()
        for sq, r in dec:
            q_use = rewards["normal"] + gamma * V[sq + r, 0]
            q_not = rewards["snake"] + gamma * V[board.snakes[sq + r], 1]
            new[sq, r] = 1 if q_use > q_not + 1e-12 else 0
        if np.array_equal(new, use):
            return use, V
        use = new
    raise RuntimeError("policy iteration did not converge")


def bellman_gap(board: Board, rewards: dict, use: np.ndarray, gamma: float) -> float:
    """Largest one-step improvement available at any decision state (0 means the policy is optimal)."""
    V = policy_value(board, rewards, use, gamma)
    gap = 0.0
    for sq, r in decision_states(board):
        q_use = rewards["normal"] + gamma * V[sq + r, 0]
        q_not = rewards["snake"] + gamma * V[board.snakes[sq + r], 1]
        better = q_use if use[sq, r] == 0 else q_not
        best = max(q_use, q_not)
        gap = max(gap, best - (q_use if use[sq, r] == 1 else q_not))
        _ = better
    return gap


def reachable_decisions(board: Board, rewards: dict, use: np.ndarray):
    """Decision states (square, roll) the game can actually reach with the shield still available,
    when playing `use` from the start. Other decision states have zero visit probability."""
    P, _ = build_chain(board, rewards, use)
    visits = np.linalg.inv(np.eye(P.shape[0]) - P)[_idx(1, 1)]
    return [(s, r) for s, r in decision_states(board) if visits[_idx(s, 1)] > 1e-9]


def fixed_policies(board: Board, rewards: dict, gamma: float) -> dict:
    n = board.n
    pol = {"never": np.zeros((n + 1, 7), dtype=int), "first_snake": np.ones((n + 1, 7), dtype=int)}
    pol["optimal"], _ = optimal_policy(board, rewards, gamma)
    return pol


# --------------------------------------------------------------------------- simulation
def simulate(board_name: str, use: np.ndarray, games: int, seed: int):
    """Play games in ShieldEnv with a fixed shield policy. Returns dict of means and standard errors."""
    env = ShieldEnv(board_name)
    rolls, rets, snakes, used, wins = [], [], [], [], 0
    for g in range(games):
        obs, _ = env.reset(seed=seed if g == 0 else None)
        done, ret, sn, us = False, 0.0, 0, 0
        while not done:
            sq, sh, r = decode(obs)
            obs, rew, term, trunc, info = env.step(int(use[sq, r]))
            ret += rew
            sn += info["event"] == "snake"
            us += info["event"] == "shielded"
            done = term or trunc
        wins += int(term)
        rolls.append(env.steps)
        rets.append(ret)
        snakes.append(sn)
        used.append(us)
    f = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a))))
    return {"win": wins / games, "rolls": f(rolls), "return": f(rets), "snakes": f(snakes), "used": f(used)}


# --------------------------------------------------------------------------- TD learning
def use_from_Q(board: Board, Q: np.ndarray) -> np.ndarray:
    """Greedy shield decision in every decision state (ties -> do not use)."""
    use = np.zeros((board.n + 1, 7), dtype=int)
    for sq, r in decision_states(board):
        o = ((sq - 1) * 2 + 1) * 6 + (r - 1)
        use[sq, r] = int(Q[o, 1] > Q[o, 0])
    return use


def td_train(board_name: str, algo: str, episodes: int, seed: int, alpha: float, gamma: float, ref: dict,
             eval_every: int, tol: float, time_limit: float | None, eps_start=1.0, eps_end=0.05, decay_frac=0.7):
    board = BOARDS[board_name]
    env = ShieldEnv(board_name)
    rng = np.random.default_rng(seed + 10_000)
    Q = np.zeros((env.observation_space.n, 2))
    dec = decision_states(board)
    opt = ref["optimal"]
    reach = ref["reach"]
    v_opt = ref["v_opt_start"]
    decay_eps = max(1, int(decay_frac * episodes))
    log, t0, stopped, ep = [], time.time(), False, 0

    def pick(o, eps):
        if rng.random() < eps:
            return int(rng.integers(0, 2))
        q = Q[o]
        return int(rng.integers(0, 2)) if q[0] == q[1] else int(q[1] > q[0])

    for ep in range(1, episodes + 1):
        eps = max(eps_end, eps_start * (eps_end / eps_start) ** (ep / decay_eps))
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a = pick(obs, eps)
        while True:
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
        if ep % eval_every == 0 or ep == episodes:
            use = use_from_Q(board, Q)
            v = policy_value(board, DEFAULT_REWARDS, use, gamma)[1, 1]
            agree = float(np.mean([use[s, r] == opt[s, r] for s, r in dec]))
            agree_r = float(np.mean([use[s, r] == opt[s, r] for s, r in reach]))
            log.append({"board": board_name, "algo": algo, "seed": seed, "episode": ep, "eps": eps,
                        "disc_value": v, "regret": v_opt - v, "agreement": agree, "agreement_reach": agree_r})
        if time_limit is not None and time.time() - t0 > time_limit:
            stopped = True
            break
    # make sure the last state of the learned policy is logged even if stopped early
    if not log or log[-1]["episode"] != ep:
        use = use_from_Q(board, Q)
        v = policy_value(board, DEFAULT_REWARDS, use, gamma)[1, 1]
        agree = float(np.mean([use[s, r] == opt[s, r] for s, r in dec]))
        agree_r = float(np.mean([use[s, r] == opt[s, r] for s, r in reach]))
        log.append({"board": board_name, "algo": algo, "seed": seed, "episode": ep, "eps": eps,
                    "disc_value": v, "regret": v_opt - v, "agreement": agree, "agreement_reach": agree_r})
    return Q, pd.DataFrame(log), stopped, time.time() - t0, ep


def stable_to_end(episodes: np.ndarray, good: np.ndarray):
    """First episode from which every later checkpoint is good (None if the last one is bad)."""
    if len(good) == 0 or not good[-1]:
        return None
    k = len(good) - 1
    while k > 0 and good[k - 1]:
        k -= 1
    return int(episodes[k])


# --------------------------------------------------------------------------- self-test
def self_test():
    # 1) with the shield not used, the rules equal the original environment, on every square and roll
    for name, b in BOARDS.items():
        orig = SmartDiceEnv(name)
        for sq in range(1, b.n):
            for r in range(1, 7):
                orig.reset()
                orig.square = sq
                obs, rew, term, _, _ = orig.step(r - 1)
                nsq, mrew, _, _ = move(b, DEFAULT_REWARDS, sq, r, 1, False)
                assert (obs, rew, term) == (nsq - 1, mrew, nsq == b.n), (name, sq, r)
    print("[ok] without using the shield the rules equal the original SmartDiceEnv, 3 boards")

    # 2) hand-checked shield cases on a tiny board: N=10, ladder 3->8, snake 6->2
    mini = Board("mini", 10, ladders={3: 8}, snakes={6: 2})
    validate_board(mini)
    R = DEFAULT_REWARDS
    assert move(mini, R, 1, 5, 1, True) == (6, -1.0, "shielded", 0)      # shield cancels the snake, stay on 6
    assert move(mini, R, 1, 5, 1, False) == (2, -20.0, "snake", 1)       # no shield: slide to 2, shield kept
    assert move(mini, R, 1, 5, 0, True) == (2, -20.0, "snake", 0)        # shield already spent: no effect
    assert move(mini, R, 1, 2, 1, True) == (8, 20.0, "ladder", 1)        # not a snake: shield NOT spent
    assert move(mini, R, 9, 6, 1, True) == (9, -10.0, "invalid", 1)      # invalid move, shield kept
    assert move(mini, R, 4, 6, 1, True) == (10, 100.0, "goal", 1)
    assert len(decision_states(mini)) == 5                               # squares 1..5 with the right roll
    e = ShieldEnv(mini)
    e.reset(seed=0)
    e.square, e.shield, e.roll = 1, 1, 5
    obs, rew, term, trunc, info = e.step(1)
    assert (e.square, e.shield, rew, info["event"]) == (6, 0, -1.0, "shielded")
    assert decode(obs)[:2] == (6, 0)
    print("[ok] hand-checked shield cases (cancel snake, no-op cases, invalid, goal) and env step")

    # 3) the DP optimum is truly optimal and never worse than the simple policies
    for name, b in BOARDS.items():
        pol = fixed_policies(b, DEFAULT_REWARDS, 0.99)
        assert bellman_gap(b, DEFAULT_REWARDS, pol["optimal"], 0.99) < 1e-9
        vals = {k: policy_value(b, DEFAULT_REWARDS, u, 0.99)[1, 1] for k, u in pol.items()}
        assert vals["optimal"] >= vals["never"] - 1e-9 and vals["optimal"] >= vals["first_snake"] - 1e-9
    print("[ok] policy iteration result is optimal (no one-step improvement) and beats never / first_snake")

    # 4) Gymnasium's checker
    from gymnasium.utils.env_checker import check_env
    for name in BOARDS:
        check_env(ShieldEnv(name), skip_render_check=True)
    print("[ok] gymnasium check_env passed on all three boards")

    # 5) simulation agrees with the exact numbers (N30, 4000 games, fixed seed)
    b = BOARDS["N30"]
    pol = fixed_policies(b, DEFAULT_REWARDS, 0.99)
    for k in ("never", "optimal"):
        ex = policy_metrics(b, DEFAULT_REWARDS, pol[k])
        mc = simulate("N30", pol[k], 4000, seed=1)
        z1 = (mc["rolls"][0] - ex["exp_rolls"]) / mc["rolls"][1]
        z2 = (mc["return"][0] - ex["exp_return"]) / mc["return"][1]
        assert abs(z1) < 4.5 and abs(z2) < 4.5, (k, z1, z2)
    print("[ok] simulated games agree with the exact analysis (N30, |z| < 4.5)")


# --------------------------------------------------------------------------- plots
def make_plots(td_log: pd.DataFrame, exact: pd.DataFrame, runs: pd.DataFrame):
    boards = list(BOARDS)
    fig, axes = plt.subplots(1, len(boards), figsize=(5 * len(boards), 4.2), squeeze=False)
    for ax, bn in zip(axes[0], boards):
        for algo in NAMES:
            d = td_log[(td_log.board == bn) & (td_log.algo == algo)]
            if d.empty:
                continue
            piv = d.pivot_table(index="episode", columns="seed", values="regret")
            ax.plot(piv.index, piv.median(axis=1), label=f"{NAMES[algo]} (median of {piv.shape[1]})")
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), alpha=0.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{bn}: regret of the greedy policy")
        ax.set_xlabel("training episodes")
        ax.set_ylabel("optimal value - learned value (discounted return)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ch2_regret.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    pols = ["never", "first_snake", "optimal"]
    x = np.arange(len(boards))
    for k, p in enumerate(pols):
        for ax, col in zip(axes, ("exp_rolls", "exp_return")):
            vals = [exact[(exact.board == b) & (exact.policy == p)][col].iloc[0] for b in boards]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=p)
    for ax, t in zip(axes, ("expected rolls per game (lower is better)", "expected return per game (higher is better)")):
        ax.set_xticks(x)
        ax.set_xticklabels(boards)
        ax.set_title(t)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ch2_policy_comparison.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Challenge 2: random dice with a one-time snake shield")
    p.add_argument("--boards", nargs="+", default=list(BOARDS), choices=list(BOARDS))
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--mc-games", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--tol", type=float, default=0.1, help="regret (discounted return) counted as 'good enough'")
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER TD run")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    self_test()

    exact_rows, mc_rows, refs = [], [], {}
    for bn in a.boards:
        b = BOARDS[bn]
        pol = fixed_policies(b, DEFAULT_REWARDS, a.gamma)
        V_opt = policy_value(b, DEFAULT_REWARDS, pol["optimal"], a.gamma)
        refs[bn] = {"optimal": pol["optimal"], "v_opt_start": float(V_opt[1, 1]),
                    "reach": reachable_decisions(b, DEFAULT_REWARDS, pol["optimal"])}
        print(f"{bn}: {len(decision_states(b))} decision states, "
              f"{len(refs[bn]['reach'])} reachable with the shield still available under the optimal policy")
        dec = decision_states(b)
        for name, use in pol.items():
            m = policy_metrics(b, DEFAULT_REWARDS, use, a.gamma)
            exact_rows.append({"board": bn, "policy": name, **m})
            mc = simulate(bn, use, a.mc_games, seed=7)
            mc_rows.append({"board": bn, "policy": name, "games": a.mc_games, "mc_win": mc["win"],
                            "exact_rolls": m["exp_rolls"], "mc_rolls": mc["rolls"][0], "se_rolls": mc["rolls"][1],
                            "z_rolls": (mc["rolls"][0] - m["exp_rolls"]) / mc["rolls"][1],
                            "exact_return": m["exp_return"], "mc_return": mc["return"][0], "se_return": mc["return"][1],
                            "z_return": (mc["return"][0] - m["exp_return"]) / mc["return"][1],
                            "exact_snakes": m["exp_snakes"], "mc_snakes": mc["snakes"][0],
                            "exact_p_used": m["p_shield_used"], "mc_p_used": mc["used"][0]})
        rows = []
        V = policy_value(b, DEFAULT_REWARDS, pol["optimal"], a.gamma)
        for sq, r in dec:
            q_use = DEFAULT_REWARDS["normal"] + a.gamma * V[sq + r, 0]
            q_not = DEFAULT_REWARDS["snake"] + a.gamma * V[b.snakes[sq + r], 1]
            rows.append({"square": sq, "roll": r, "snake_head": sq + r, "snake_tail": b.snakes[sq + r],
                         "use_shield_optimal": int(pol["optimal"][sq, r]), "q_use_minus_q_not": q_use - q_not})
        pd.DataFrame(rows).to_csv(OUT / f"ch2_optimal_policy_{bn}.csv", index=False)

    exact = pd.DataFrame(exact_rows)
    mc_df = pd.DataFrame(mc_rows)
    exact.to_csv(OUT / "ch2_exact_policies.csv", index=False)
    mc_df.to_csv(OUT / "ch2_mc_check.csv", index=False)
    print("\nEXACT ANALYSIS (infinite-horizon Markov chain, undiscounted game statistics; "
          f"optimal = best discounted return, gamma {a.gamma})")
    print(exact.round(3).to_string(index=False))
    print("\nSHIELD VALUE (optimal vs never use): the benefit of the special action")
    for bn in a.boards:
        e = exact[exact.board == bn].set_index("policy")
        print(f"  {bn}: expected return {e.loc['never', 'exp_return']:.2f} -> {e.loc['optimal', 'exp_return']:.2f} "
              f"(+{e.loc['optimal', 'exp_return'] - e.loc['never', 'exp_return']:.2f}), expected rolls "
              f"{e.loc['never', 'exp_rolls']:.2f} -> {e.loc['optimal', 'exp_rolls']:.2f}, "
              f"P(shield spent) under optimal = {e.loc['optimal', 'p_shield_used']:.3f}")
    print("\nSIMULATION vs EXACT (z = (simulated - exact) / standard error)")
    print(mc_df[["board", "policy", "games", "exact_rolls", "mc_rolls", "z_rolls", "exact_return", "mc_return",
                 "z_return", "exact_p_used", "mc_p_used", "mc_win"]].round(3).to_string(index=False))
    print("\nWHERE THE OPTIMAL POLICY USES THE SHIELD (decision states = square+roll lands on a snake head)")
    for bn in a.boards:
        d = pd.read_csv(OUT / f"ch2_optimal_policy_{bn}.csv")
        g = d.groupby(["snake_head", "snake_tail"]).agg(states=("square", "count"), use=("use_shield_optimal", "sum"))
        g["drop"] = g.index.get_level_values(0) - g.index.get_level_values(1)
        print(f"  {bn}: uses the shield in {int(d.use_shield_optimal.sum())} of {len(d)} decision states")
        print(g.reset_index().to_string(index=False))

    # --- TD learning
    run_rows, logs = [], []
    for bn in a.boards:
        for algo in NAMES:
            for seed in a.seeds:
                Q, log, stopped, secs, ep = td_train(bn, algo, a.episodes, seed, a.alpha, a.gamma, refs[bn],
                                                     a.eval_every, a.tol, a.time_limit)
                logs.append(log)
                good = (log["regret"] <= a.tol).to_numpy()
                conv = stable_to_end(log["episode"].to_numpy(), good)
                use = use_from_Q(BOARDS[bn], Q)
                m = policy_metrics(BOARDS[bn], DEFAULT_REWARDS, use, a.gamma)
                last = log.iloc[-1]
                run_rows.append({"board": bn, "algo": algo, "seed": seed, "episodes_run": ep, "stopped": stopped,
                                 "elapsed_s": round(secs, 1), "final_regret": last["regret"],
                                 "final_agreement": last["agreement"], "final_agreement_reach": last["agreement_reach"],
                                 "conv_episode": conv,
                                 "exp_rolls": m["exp_rolls"], "exp_return": m["exp_return"],
                                 "p_shield_used": m["p_shield_used"]})
                print(f"[{algo} {bn} seed{seed}] {ep} episodes in {secs:.1f}s | final regret {last['regret']:.3f} | "
                      f"agreement with optimal {last['agreement']:.2f} (reachable states {last['agreement_reach']:.2f}) | "
                      f"{'converged at episode ' + str(conv) if conv else 'NOT converged'}"
                      f"{' | STOPPED by time limit' if stopped else ''}", flush=True)
    td_log = pd.concat(logs, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    td_log.to_csv(OUT / "ch2_td_log.csv", index=False)
    runs.to_csv(OUT / "ch2_td_runs.csv", index=False)
    summ = (runs.groupby(["board", "algo"], sort=False)
            .agg(n=("seed", "count"), final_regret_median=("final_regret", "median"),
                 final_regret_max=("final_regret", "max"), agreement_median=("final_agreement", "median"),
                 agreement_reach_median=("final_agreement_reach", "median"),
                 not_converged=("conv_episode", lambda s: int(s.isna().sum())),
                 conv_episode_median=("conv_episode", "median"),
                 exp_return_median=("exp_return", "median")).round(3).reset_index())
    summ.to_csv(OUT / "ch2_td_summary.csv", index=False)
    print(f"\nTD LEARNING SUMMARY ({a.episodes} episodes, alpha {a.alpha}, tolerance {a.tol} regret). "
          "Regret = optimal discounted value - value of the learned greedy policy (exact).")
    print(summ.to_string(index=False))
    make_plots(td_log, exact, runs)
    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
