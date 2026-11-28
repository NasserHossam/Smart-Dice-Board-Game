"""
challenge4_two_players.py - CHALLENGE 4: the agent races a rule-based opponent.

DESIGN (my choices; the report should say so)
  * Two players on the same board take turns. The AGENT chooses its dice value (as in the original game).
    The OPPONENT follows a rule. First to reach square N wins. Rewards for the agent: +1 win, -1 loss,
    0 otherwise (the objective is the probability of winning, so I use it directly).
  * Who moves first: --first agent | opponent | random. If the opponent moves first, its first move is made
    inside reset(), so the agent always acts at its own turn.
  * State = (agent square, opponent square), Discrete(N*N), obs = (agent-1)*N + (opponent-1).
  * Opponents (all use the same rules as the agent for ladders, snakes and invalid moves):
      random : uniformly random roll every turn
      rule   : hand-written rule from the local view (goal > ladder > normal > invalid > snake, ties ->
               larger roll), but with probability --opp-eps (default 0.3) it rolls randomly instead
      bfs    : always follows a shortest path to the goal (deterministic, the strongest possible speed)
  * Two game variants:
      no bump (default): players do not affect each other. This is the plain race.
      --bump  : my own addition, NOT part of the original rules. If a player ends a move on the square the
               other player stands on, the other player goes back to square 1.
  * Exact analysis: with a fixed opponent and a fixed agent policy, the game is a Markov chain over
    (agent square, opponent square). The exact win probability is found by iterating the Bellman equations
    to convergence (no sampling noise). The OPTIMAL agent is found by value iteration on the win probability.
    Compared agents: random rolls, BFS shortest path (ignores the opponent), optimal, and the learned ones.
  * THEORY CHECK built into the self-test: without bump the players do not interact, so the win probability
    only depends on how many rolls the agent needs. The BFS agent is then already optimal and the opponent's
    position carries no useful information. With bump, that can change.
  * Q-Learning (and SARSA with --algos) learn on the N*N states from win/loss rewards only. Their greedy
    policies are scored exactly. gamma = 1 (episodic win/loss), alpha 0.1, epsilon 1.0 -> 0.05 over 70%.

Usage (pilot first):
  python challenge4_two_players.py --episodes 4000 --seeds 0 --mc-games 500 --boards N30
  python challenge4_two_players.py --episodes 40000 --seeds 0 1 2
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

from env import BOARDS, Board, next_square, optimal_roll_sets, validate_board

OUT = Path(__file__).parent / "results_challenges"
OPPONENTS = ("random", "rule", "bfs")
INVALID, SNAKE, NORMAL, LADDER, GOAL = 0, 1, 2, 3, 4


# --------------------------------------------------------------------------- board helpers
def move_table(board: Board) -> np.ndarray:
    """mv[sq, roll] = square after the roll (stays on sq if the move is invalid)."""
    mv = np.zeros((board.n + 1, 7), dtype=int)
    for sq in range(1, board.n):
        for r in range(1, 7):
            nx = next_square(board, sq, r)
            mv[sq, r] = sq if nx is None else nx
    return mv


def local_rule_roll(board: Board, sq: int) -> int:
    """Rule using only what the next 6 rolls would do: goal > ladder > normal > invalid > snake, ties -> larger roll."""
    prio = {GOAL: 5, LADDER: 4, NORMAL: 3, INVALID: 2, SNAKE: 1}

    def cat(r):
        t = sq + r
        if t > board.n:
            return INVALID
        if board.jump_map.get(t, t) == board.n:
            return GOAL
        if t in board.ladders:
            return LADDER
        if t in board.snakes:
            return SNAKE
        return NORMAL

    return max(range(1, 7), key=lambda r: (prio[cat(r)], r))


def opponent_mu(board: Board, kind: str, eps: float) -> np.ndarray:
    """mu[sq, k] = probability that the opponent rolls k+1 when standing on sq."""
    n = board.n
    mu = np.full((n + 1, 6), 1 / 6)
    if kind == "random":
        return mu
    sets = optimal_roll_sets(board)
    e = eps if kind == "rule" else 0.0
    for sq in range(1, n):
        best = local_rule_roll(board, sq) if kind == "rule" else min(sets[sq])
        mu[sq] = e / 6
        mu[sq, best - 1] += 1 - e
    return mu


# --------------------------------------------------------------------------- environment
class RaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, board: Board | str = "N30", opponent: str = "random", opp_eps: float = 0.3,
                 bump: bool = False, first: str = "agent", max_rounds: int | None = None):
        super().__init__()
        self.board = BOARDS[board] if isinstance(board, str) else board
        validate_board(self.board)
        assert opponent in OPPONENTS and first in ("agent", "opponent", "random")
        self.n, self.bump, self.first = self.board.n, bump, first
        self.mv = move_table(self.board)
        self.mu = opponent_mu(self.board, opponent, opp_eps)
        self.cum = np.cumsum(self.mu, axis=1)
        self.max_rounds = int(max_rounds) if max_rounds is not None else 5 * self.n
        self.observation_space = spaces.Discrete(self.n * self.n)
        self.action_space = spaces.Discrete(6)
        self.a, self.o, self.rounds = 1, 1, 0

    def _obs(self) -> int:
        return int((self.a - 1) * self.n + (self.o - 1))

    def _opponent_move(self) -> bool:
        """Opponent takes its turn. Returns True if it reached the goal."""
        u = float(self.np_random.random())
        k = min(int(np.searchsorted(self.cum[self.o], u, side="right")), 5)
        no = int(self.mv[self.o, k + 1])
        if no == self.n:
            self.o = self.n
            return True
        if self.bump and no == self.a:
            self.a = 1
        self.o = no
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.a, self.o, self.rounds = 1, 1, 0
        opp_first = self.first == "opponent" or (self.first == "random" and self.np_random.random() < 0.5)
        if opp_first:
            assert not self._opponent_move(), "opponent cannot win on its first move on these boards"
        return self._obs(), {"opponent_first": bool(opp_first)}

    def step(self, action):
        action = int(action)
        if not 0 <= action <= 5:
            raise ValueError(f"action must be in 0..5, got {action}")
        self.rounds += 1
        na = int(self.mv[self.a, action + 1])
        if na == self.n:
            self.a = self.n
            return self._obs(), 1.0, True, False, {"result": "win"}
        if self.bump and na == self.o:
            self.o = 1
        self.a = na
        if self._opponent_move():
            return self._obs(), -1.0, True, False, {"result": "loss"}
        return self._obs(), 0.0, False, self.rounds >= self.max_rounds, {}


# --------------------------------------------------------------------------- exact analysis
class GameModel:
    """Markov model of the race for one (board, opponent, bump). Agent-to-move states (a, o), both < N.
    Q_r(s) = P(agent wins | state s, agent rolls r) = win_r + sum_r2 coef[r, r2] * W[idx[r, r2]]."""

    def __init__(self, board: Board, opponent: str, opp_eps: float, bump: bool):
        n = board.n
        self.board, self.n, self.m, self.bump = board, n, n - 1, bump
        self.S = self.m * self.m
        mv, mu = move_table(board), opponent_mu(board, opponent, opp_eps)
        self.mu, self.mv = mu, mv
        self.idx = np.zeros((6, 6, self.S), dtype=int)
        self.coef = np.zeros((6, 6, self.S))
        self.win = np.zeros((6, self.S))
        for a in range(1, n):
            for o in range(1, n):
                s = (a - 1) * self.m + (o - 1)
                for r in range(1, 7):
                    na = int(mv[a, r])
                    if na == n:
                        self.win[r - 1, s] = 1.0
                        continue
                    oa = 1 if (bump and na == o) else o
                    for r2 in range(1, 7):
                        p = mu[oa, r2 - 1]
                        if p == 0:
                            continue
                        no = int(mv[oa, r2])
                        if no == n:
                            continue                                    # opponent wins: contributes 0
                        nna = 1 if (bump and no == na) else na
                        self.idx[r - 1, r2 - 1, s] = (nna - 1) * self.m + (no - 1)
                        self.coef[r - 1, r2 - 1, s] = p

    def q(self, W):
        Q = self.win.copy()
        for r in range(6):
            for r2 in range(6):
                Q[r] += self.coef[r, r2] * W[self.idx[r, r2]]
        return Q                                                        # shape (6, S)

    def evaluate(self, pi, W0=None, tol=1e-11, max_sweeps=60000):
        """Exact win probability of the agent policy pi (S, 6 probabilities)."""
        W = np.zeros(self.S) if W0 is None else W0.copy()
        for _ in range(max_sweeps):
            Wn = (pi.T * self.q(W)).sum(axis=0)
            if np.max(np.abs(Wn - W)) < tol:
                return Wn
            W = Wn
        return W

    def optimal(self, tol=1e-11, max_sweeps=60000):
        """Optimal agent. Value iteration gives the optimal win probability W*; the policy is then found by
        policy iteration starting from the BFS agent (a policy that always finishes). Ties keep the current
        action, so the final policy never loops where a finishing action exists. Returns (W of that policy, pi)."""
        W = np.zeros(self.S)
        for _ in range(max_sweeps):
            Wn = self.q(W).max(axis=0)
            if np.max(np.abs(Wn - W)) < tol:
                W = Wn
                break
            W = Wn
        W_vi = W
        pi = pi_bfs(self)
        act = pi.argmax(axis=1)
        Wp = self.evaluate(pi)
        for _ in range(200):
            Q = self.q(Wp)
            cur = Q[act, np.arange(self.S)]
            best = Q.max(axis=0)
            new_act = np.where(best > cur + 1e-12, Q.argmax(axis=0), act)
            if np.array_equal(new_act, act):
                break
            act = new_act
            pi = np.zeros((self.S, 6))
            pi[np.arange(self.S), act] = 1.0
            Wp = self.evaluate(pi, Wp)
        if np.max(np.abs(Wp - W_vi)) > 1e-6:
            raise RuntimeError("policy iteration and value iteration disagree")
        return Wp, pi

    def start_value(self, W, first: str) -> float:
        """Win probability at the start of a game for who-moves-first mode."""
        s0 = 0                                                          # agent 1, opponent 1
        agent_first = float(W[s0])
        opp_first = 0.0
        for r2 in range(1, 7):
            p = self.mu[1, r2 - 1]
            no = int(self.mv[1, r2])
            if p == 0 or no == self.n:
                continue
            opp_first += p * W[0 * self.m + (no - 1)]                   # agent stays on 1 (bump onto 1 changes nothing)
        return {"agent": agent_first, "opponent": float(opp_first), "random": 0.5 * (agent_first + opp_first)}[first]


def pi_random(model: GameModel) -> np.ndarray:
    return np.full((model.S, 6), 1 / 6)


def pi_bfs(model: GameModel) -> np.ndarray:
    sets = optimal_roll_sets(model.board)
    pi = np.zeros((model.S, 6))
    for a in range(1, model.n):
        pi[(a - 1) * model.m:(a - 1) * model.m + model.m, min(sets[a]) - 1] = 1.0
    return pi


def q_to_pi(Q: np.ndarray, n: int) -> np.ndarray:
    """Greedy policy from a learned table over the N*N observations; ties -> uniform over the tied rolls."""
    m = n - 1
    sub = Q.reshape(n, n, 6)[:m, :m].reshape(m * m, 6)
    best = sub == sub.max(axis=1, keepdims=True)
    return best / best.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- simulation
def simulate(board_name, opponent, opp_eps, bump, first, pi, games, seed):
    env = RaceEnv(board_name, opponent, opp_eps, bump, first)
    n, m = env.n, env.n - 1
    cum = np.cumsum(pi, axis=1)
    rng = np.random.default_rng(seed)
    wins = 0
    for g in range(games):
        obs, _ = env.reset(seed=seed if g == 0 else None)
        done = False
        while not done:
            a0, o0 = divmod(int(obs), n)
            row = cum[a0 * m + o0]
            act = min(int(np.searchsorted(row, rng.random() * row[-1], side="right")), 5)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        wins += info.get("result") == "win"
    p = wins / games
    return p, float(np.sqrt(max(p * (1 - p), 1e-12) / games))


# --------------------------------------------------------------------------- TD learning
def td_train(board_name, opponent, opp_eps, bump, first, algo, episodes, seed, alpha, gamma, model,
             W_opt_start, n_evals, time_limit, eps_start=1.0, eps_end=0.05, decay_frac=0.7):
    env = RaceEnv(board_name, opponent, opp_eps, bump, first)
    n = env.n
    rng = np.random.default_rng(seed + 4242)
    Q = np.zeros((n * n, 6))
    decay_eps = max(1, int(decay_frac * episodes))
    checkpoints = set(int(round(episodes * k / n_evals)) for k in range(1, n_evals + 1))
    log, t0, W_prev, ep = [], time.time(), None, 0

    def pick(o, eps):
        if rng.random() < eps:
            return int(rng.integers(0, 6))
        q = Q[o]
        idx = np.flatnonzero(q == q.max())
        return int(idx[0]) if len(idx) == 1 else int(rng.choice(idx))

    def checkpoint(ep, eps):
        nonlocal W_prev
        W_prev = model.evaluate(q_to_pi(Q, n), W_prev)
        v = {f: model.start_value(W_prev, f) for f in ("agent", "opponent", "random")}
        log.append({"episode": ep, "eps": eps, "win_prob": v[first], "regret": W_opt_start - v[first],
                    "win_agent_first": v["agent"], "win_opp_first": v["opponent"]})

    eps = eps_start
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
        if ep in checkpoints:
            checkpoint(ep, eps)
        if time_limit is not None and time.time() - t0 > time_limit:
            if not log or log[-1]["episode"] != ep:
                checkpoint(ep, eps)
            break
    return Q, pd.DataFrame(log), time.time() - t0, ep


# --------------------------------------------------------------------------- self-test
def self_test():
    from env import SmartDiceEnv

    # 1) move table equals the original environment
    for name, b in BOARDS.items():
        mv, orig = move_table(b), SmartDiceEnv(name)
        for sq in range(1, b.n):
            for r in range(1, 7):
                orig.reset()
                orig.square = sq
                obs, _, _, _, _ = orig.step(r - 1)
                assert obs == mv[sq, r] - 1, (name, sq, r)
    print("[ok] move rules equal the original SmartDiceEnv for every square and roll, 3 boards")

    # 2) hand-checkable results on N30 (both players need 4 rolls with the BFS rule), no bump
    b = BOARDS["N30"]
    mdl = GameModel(b, "bfs", 0.0, False)
    Wb = mdl.evaluate(pi_bfs(mdl))
    got = [mdl.start_value(Wb, f) for f in ("agent", "opponent", "random")]
    assert np.allclose(got, [1.0, 0.0, 0.5], atol=1e-9), got
    print("[ok] BFS agent vs BFS opponent on N30: wins 1.0 if it moves first, 0.0 if second, 0.5 with a coin flip")

    # 3) independent check of the joint model: BFS agent vs random opponent, opponent moves first.
    #    The agent needs 4 rolls, so it wins iff the opponent has NOT finished in its first 4 moves.
    mu = opponent_mu(b, "random", 0.0)
    mv = move_table(b)
    dist = np.zeros(b.n + 1)
    dist[1] = 1.0
    finished = 0.0
    for _ in range(4):
        nd = np.zeros(b.n + 1)
        for sq in range(1, b.n):
            for r in range(1, 7):
                nd[mv[sq, r]] += dist[sq] / 6
        finished += nd[b.n]
        nd[b.n] = 0.0
        dist = nd
    mdl_r = GameModel(b, "random", 0.0, False)
    W = mdl_r.evaluate(pi_bfs(mdl_r))
    assert abs(mdl_r.start_value(W, "opponent") - (1 - finished)) < 1e-9, (mdl_r.start_value(W, "opponent"), 1 - finished)
    print("[ok] joint model equals an independent one-player calculation (BFS agent vs random opponent, N30)")

    # 4) optimal >= other agents everywhere; without bump the BFS agent is already optimal (theory check)
    for name, b in BOARDS.items():
        for kind in OPPONENTS:
            for bump in (False, True):
                if name == "N100" and bump:
                    continue
                mdl = GameModel(b, kind, 0.3, bump)
                Wo, _ = mdl.optimal()
                Wr = mdl.evaluate(pi_random(mdl))
                Wf = mdl.evaluate(pi_bfs(mdl))
                assert np.all(Wo >= Wr - 1e-9) and np.all(Wo >= Wf - 1e-9), (name, kind, bump)
                if not bump:
                    assert np.max(np.abs(Wo - Wf)) < 1e-8, (name, kind)
    print("[ok] optimal agent beats random and BFS agents in every state; without bump the BFS agent is optimal")

    # 5) Gymnasium's checker, both variants
    from gymnasium.utils.env_checker import check_env
    for bump in (False, True):
        check_env(RaceEnv("N30", "rule", 0.3, bump, "random"), skip_render_check=True)
    print("[ok] gymnasium check_env passed (with and without bump, random first mover)")

    # 6) simulated games agree with the exact analysis (N30, fixed seeds)
    for kind in OPPONENTS:
        for bump in (False, True):
            mdl = GameModel(BOARDS["N30"], kind, 0.3, bump)
            for nm, pi in (("random", pi_random(mdl)), ("bfs", pi_bfs(mdl))):
                ex = mdl.start_value(mdl.evaluate(pi), "random")
                p, se = simulate("N30", kind, 0.3, bump, "random", pi, 4000, seed=3)
                assert abs(p - ex) < 4.5 * max(se, 1e-3), (kind, bump, nm, p, ex)
    print("[ok] simulated games agree with the exact win probabilities (N30, |z| < 4.5)")


# --------------------------------------------------------------------------- plots
def make_plots(exact: pd.DataFrame, td_runs: pd.DataFrame, td_log: pd.DataFrame, first: str):
    boards = list(dict.fromkeys(exact.board))
    fig, axes = plt.subplots(2, len(boards), figsize=(5.2 * len(boards), 7.5), squeeze=False)
    for r, bump in enumerate((False, True)):
        for c, bn in enumerate(boards):
            ax = axes[r][c]
            d = exact[(exact.board == bn) & (exact.bump == bump)]
            if d.empty:
                ax.axis("off")
                continue
            x = np.arange(len(OPPONENTS))
            for k, (col, lab) in enumerate((("random_agent", "random rolls"), ("bfs_agent", "BFS shortest path"),
                                            ("optimal_agent", "optimal"), ("learned_median", "learned (median)"))):
                vals = [d[d.opponent == o][col].iloc[0] if (d.opponent == o).any() else np.nan for o in OPPONENTS]
                ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels(OPPONENTS)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{bn}, {'with bump' if bump else 'no bump'} (agent moves {first} first)")
            ax.set_ylabel("exact win probability")
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"ch4_win_probability_{first}_first.png", dpi=130)
    plt.close(fig)

    if not td_log.empty:
        fig, axes = plt.subplots(2, len(boards), figsize=(5.2 * len(boards), 7.5), squeeze=False)
        for r, bump in enumerate((False, True)):
            for c, bn in enumerate(boards):
                ax = axes[r][c]
                for o in OPPONENTS:
                    d = td_log[(td_log.board == bn) & (td_log.bump == bump) & (td_log.opponent == o)]
                    if d.empty:
                        continue
                    piv = d.pivot_table(index="episode", columns="seed", values="regret")
                    ax.plot(piv.index, piv.median(axis=1), marker="o", label=f"vs {o}")
                ax.axhline(0, color="k", lw=0.8)
                ax.set_title(f"{bn}, {'with bump' if bump else 'no bump'}")
                ax.set_xlabel("training episodes")
                ax.set_ylabel("optimal win prob - learned win prob")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"ch4_regret_{first}_first.png", dpi=130)
        plt.close(fig)


def med_rng(x, nd=3):
    x = pd.Series(x).dropna()
    return "n/a" if x.empty else f"{x.median():.{nd}f} [{x.min():.{nd}f}-{x.max():.{nd}f}]"


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Challenge 4: agent vs a rule-based opponent")
    p.add_argument("--boards", nargs="+", default=["N30", "N50"], choices=list(BOARDS))
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=list(OPPONENTS))
    p.add_argument("--opp-eps", type=float, default=0.3, help="randomness of the 'rule' opponent")
    p.add_argument("--first", default="agent", choices=["agent", "opponent", "random"])
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--algos", nargs="+", default=["qlearning"], choices=["qlearning", "sarsa"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--mc-games", type=int, default=4000)
    p.add_argument("--n-evals", type=int, default=8, help="exact evaluations of the learned policy per run")
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER training run")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    sfx = f"_{a.first}_first"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    self_test()

    exact_rows, mc_rows, run_rows, logs = [], [], [], []
    for bn in a.boards:
        board = BOARDS[bn]
        for kind in a.opponents:
            for bump in (False, True):
                t0 = time.time()
                model = GameModel(board, kind, a.opp_eps, bump)
                Wo, pi_o = model.optimal()
                pols = {"random_agent": pi_random(model), "bfs_agent": pi_bfs(model), "optimal_agent": pi_o}
                Ws = {k: (Wo if k == "optimal_agent" else model.evaluate(v)) for k, v in pols.items()}
                row = {"board": bn, "opponent": kind, "bump": bump}
                for f in ("agent", "opponent", "random"):
                    for k in pols:
                        row[f"{k}_first_{f}"] = model.start_value(Ws[k], f)
                for k in pols:
                    row[k] = row[f"{k}_first_{a.first}"]
                exact_rows.append(row)
                for k, pi in pols.items():
                    ps, se = simulate(bn, kind, a.opp_eps, bump, a.first, pi, a.mc_games, seed=11)
                    mc_rows.append({"board": bn, "opponent": kind, "bump": bump, "agent": k, "exact": row[k],
                                    "simulated": ps, "se": se, "z": (ps - row[k]) / max(se, 1e-9)})
                W_opt_start = row["optimal_agent"]
                for algo in a.algos:
                    for seed in a.seeds:
                        Q, log, secs, ep = td_train(bn, kind, a.opp_eps, bump, a.first, algo, a.episodes, seed,
                                                    a.alpha, a.gamma, model, W_opt_start, a.n_evals, a.time_limit)
                        log = log.assign(board=bn, opponent=kind, bump=bump, algo=algo, seed=seed)
                        logs.append(log)
                        last = log.iloc[-1]
                        run_rows.append({"board": bn, "opponent": kind, "bump": bump, "algo": algo, "seed": seed,
                                         "episodes_run": ep, "elapsed_s": round(secs, 1),
                                         "win_prob": last["win_prob"], "regret": last["regret"],
                                         "optimal": W_opt_start, "bfs_agent": row["bfs_agent"],
                                         "random_agent": row["random_agent"]})
                        print(f"[{algo} {bn} vs {kind} {'bump' if bump else 'no-bump'} seed{seed}] {ep} ep, {secs:.1f}s | "
                              f"win prob {last['win_prob']:.3f} (optimal {W_opt_start:.3f}, BFS agent {row['bfs_agent']:.3f}, "
                              f"random agent {row['random_agent']:.3f})", flush=True)
                print(f"   ... {bn} vs {kind} {'bump' if bump else 'no-bump'} done in {time.time() - t0:.0f}s", flush=True)

    runs = pd.DataFrame(run_rows)
    exact = pd.DataFrame(exact_rows)
    med = runs.groupby(["board", "opponent", "bump"]).win_prob.median().rename("learned_median").reset_index()
    exact = exact.merge(med, on=["board", "opponent", "bump"], how="left")
    td_log = pd.concat(logs, ignore_index=True)
    mc_df = pd.DataFrame(mc_rows)
    exact.to_csv(OUT / f"ch4_exact{sfx}.csv", index=False)
    mc_df.to_csv(OUT / f"ch4_mc_check{sfx}.csv", index=False)
    runs.to_csv(OUT / f"ch4_td_runs{sfx}.csv", index=False)
    td_log.to_csv(OUT / f"ch4_td_log{sfx}.csv", index=False)

    show = exact[["board", "opponent", "bump", "random_agent", "bfs_agent", "optimal_agent",
                  "optimal_agent_first_agent", "optimal_agent_first_opponent"]].copy()
    show["gain_of_optimal_over_bfs"] = show.optimal_agent - show.bfs_agent
    print(f"\nEXACT WIN PROBABILITY OF THE AGENT (agent moves {a.first} first; last two columns of the table above "
          "use agent-first and opponent-first)")
    print(show.round(4).to_string(index=False))
    print("\nSIMULATION vs EXACT (z = (simulated - exact) / standard error)")
    print(mc_df.round(3).to_string(index=False))

    summ = []
    for (bn, kind, bump, algo), g in runs.groupby(["board", "opponent", "bump", "algo"], sort=False):
        summ.append({"board": bn, "opponent": kind, "bump": bump, "algo": algo, "n_seeds": len(g),
                     "win_prob": med_rng(g.win_prob), "regret": med_rng(g.regret),
                     "optimal": round(g.optimal.iloc[0], 3), "bfs_agent": round(g.bfs_agent.iloc[0], 3),
                     "random_agent": round(g.random_agent.iloc[0], 3),
                     "runs_at_least_BFS_agent": int((g.win_prob >= g.bfs_agent - 1e-6).sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / f"ch4_td_summary{sfx}.csv", index=False)
    print(f"\nTD LEARNING ({a.episodes} episodes, alpha {a.alpha}, gamma {a.gamma}). win_prob = exact win probability "
          "of the learned greedy policy, median [min-max] over seeds")
    print(summ.to_string(index=False))
    make_plots(exact, runs, td_log, a.first)
    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
