"""
tabular.py - GROUP B: tabular Q-Learning and SARSA written with NumPy only.

Both methods keep a table Q[state, action] with shape (N, 6) and use epsilon-greedy exploration.

  Q-Learning (off-policy): Q(s,a) += alpha * (r + gamma * max_a' Q(s',a')   - Q(s,a))
  SARSA      (on-policy) : Q(s,a) += alpha * (r + gamma * Q(s',a')          - Q(s,a))
                           where a' is the action epsilon-greedy ACTUALLY picks next.

Details (write these in the report):
  * terminal step (goal reached): target = r, no bootstrapping.
  * a time-limit truncation is NOT terminal, so the update still bootstraps.
  * epsilon decays exponentially per episode from eps_start to eps_end, reaching eps_end after
    eps_decay_frac * episodes episodes, then stays at eps_end.
  * Q starts at 0. Because every normal step costs -1, untried actions look better than tried
    ones, which pushes the agent to explore (optimistic initialisation). Mention this.
  * the environment is deterministic, so a greedy policy plays the same game every time;
    evaluation ties between equal Q-values are broken randomly, which is the only variation.

CONVERGENCE CRITERION (used in the report):
  At a checkpoint the greedy policy plays eval_games games (default 100). Checkpoints happen after
  EVERY episode for the first eval_dense_until episodes (default 50), then every eval_every episodes.
  A checkpoint is "good" if success rate >= 95% AND mean rolls <= 1.10 * BFS optimum.
  Converged at episode E if E is good AND every checkpoint in [E, E + stable_window] is good
  (default window = 50 episodes) AND the run lasted at least until E + stable_window.
  If this never happens, the run is reported as NOT converged.
  A second, stricter number is also stored: optimal_episode = same rule but with mean rolls
  EQUAL to the BFS optimum (the 10% rule can accept a policy that is 1 roll slower on N100).

Example (small pilot first!):
  python src/tabular.py --board N30 --algo both --episodes 300 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from env import BOARDS, SmartDiceEnv, bfs_min_rolls

from paths import TABULAR

RESULTS_DIR = TABULAR   # results/tabular/
ALGOS = ("qlearning", "sarsa")


# --------------------------------------------------------------------------- action selection
def greedy_action(q_row: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def epsilon_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, 6))
    return greedy_action(q_row, rng)


# --------------------------------------------------------------------------- evaluation
def evaluate_greedy(env: SmartDiceEnv, Q: np.ndarray, rng: np.random.Generator, games: int = 100):
    """Play `games` games with the greedy policy. Returns (success_rate, mean_rolls, mean_return)."""
    wins, moves, rets = 0, [], []
    for _ in range(games):
        obs, _ = env.reset()
        done, ret, term = False, 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(greedy_action(Q[obs], rng))
            ret += r
            done = term or trunc
        wins += int(term)
        moves.append(env.steps)
        rets.append(ret)
    return wins / games, float(np.mean(moves)), float(np.mean(rets))


def meets_criterion(success: float, moves: float, bfs: int) -> bool:
    return success >= 0.95 and moves <= 1.10 * bfs


def meets_optimal(success: float, moves: float, bfs: int) -> bool:
    return success >= 0.95 and moves <= bfs + 1e-9


def find_stable(eval_rows: list, ok_fn, bfs: int, window: int = 50):
    """eval_rows: list of (episode, success, moves, return), in episode order.
    Returns the first episode E where ok_fn holds at every checkpoint in [E, E+window],
    provided the run reached E+window. Returns None otherwise (= NOT converged)."""
    last_ep = eval_rows[-1][0]
    for i, (ep, s, m, _) in enumerate(eval_rows):
        if last_ep < ep + window:
            break
        if ok_fn(s, m, bfs) and all(ok_fn(s2, m2, bfs) for e2, s2, m2, _ in eval_rows[i:] if e2 <= ep + window):
            return int(ep)
    return None


# --------------------------------------------------------------------------- training
def train(board: str, algo: str, episodes: int, seed: int,
          alpha: float = 0.1, gamma: float = 0.99,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_frac: float = 0.7,
          eval_every: int = 10, eval_games: int = 100,
          eval_dense_until: int = 50, stable_window: int = 50,
          time_limit: float | None = None, rewards: dict | None = None,
          max_steps: int | None = None, verbose: bool = True):
    """Train one agent. Returns (Q, log_dataframe, summary_dict)."""
    assert algo in ALGOS, f"algo must be one of {ALGOS}"
    rng = np.random.default_rng(seed)                 # exploration randomness
    rng_eval = np.random.default_rng(seed + 10_000)   # tie-breaking in evaluation only
    env = SmartDiceEnv(board, rewards=rewards, max_steps=max_steps)
    env_eval = SmartDiceEnv(board, rewards=rewards, max_steps=max_steps)
    Q = np.zeros((env.n, 6))
    bfs = bfs_min_rolls(env.board)
    decay = (eps_end / eps_start) ** (1.0 / max(1, int(eps_decay_frac * episodes)))
    eps = eps_start

    ep_ret, ep_moves, ep_succ, ep_eps = [], [], [], []
    eval_rows = []
    t0 = time.time()
    stopped_by_time = False

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a = epsilon_greedy(Q[obs], eps, rng)
        ret, done, term = 0.0, False, False
        while not done:
            nobs, r, term, trunc, _ = env.step(a)
            ret += r
            na = epsilon_greedy(Q[nobs], eps, rng)
            if term:
                target = r
            elif algo == "sarsa":
                target = r + gamma * Q[nobs, na]
            else:
                target = r + gamma * Q[nobs].max()
            Q[obs, a] += alpha * (target - Q[obs, a])
            obs, a = nobs, na
            done = term or trunc

        ep_ret.append(ret)
        ep_moves.append(env.steps)
        ep_succ.append(int(term))
        ep_eps.append(eps)
        eps = max(eps_end, eps * decay)

        out_of_time = time_limit is not None and (time.time() - t0) > time_limit
        if ep <= eval_dense_until or ep % eval_every == 0 or ep == episodes or out_of_time:
            s, m, g = evaluate_greedy(env_eval, Q, rng_eval, eval_games)
            eval_rows.append((ep, s, m, g))
            if verbose and (ep % max(eval_every, episodes // 10) == 0 or ep == episodes or out_of_time):
                print(f"  [{algo} {board} seed{seed}] ep {ep:>6}/{episodes}  eps={ep_eps[-1]:.3f}  "
                      f"greedy: success={s * 100:5.1f}%  rolls={m:6.2f} (BFS {bfs})  "
                      f"return={g:7.1f}  t={time.time() - t0:5.1f}s", flush=True)
        if out_of_time:
            stopped_by_time = True
            break

    episodes_run = len(ep_ret)
    log = pd.DataFrame({"episode": np.arange(1, episodes_run + 1), "epsilon": ep_eps,
                        "train_return": ep_ret, "train_moves": ep_moves, "train_success": ep_succ})
    ev = pd.DataFrame(eval_rows, columns=["episode", "eval_success", "eval_moves", "eval_return"])
    log = log.merge(ev, on="episode", how="left")

    conv = find_stable(eval_rows, meets_criterion, bfs, stable_window)
    opt = find_stable(eval_rows, meets_optimal, bfs, stable_window)
    last = eval_rows[-1]
    summary = {
        "algo": algo, "board": board, "seed": seed, "bfs_optimum": bfs,
        "episodes_planned": episodes, "episodes_run": episodes_run,
        "stopped_by_time_limit": stopped_by_time, "elapsed_s": round(time.time() - t0, 2),
        "converged": conv is not None, "convergence_episode": conv, "optimal_episode": opt,
        "final_eval_success": last[1], "final_eval_moves": last[2], "final_eval_return": last[3],
        "alpha": alpha, "gamma": gamma, "eps_start": eps_start, "eps_end": eps_end,
        "eps_decay_frac": eps_decay_frac, "eval_every": eval_every, "eval_games": eval_games,
        "eval_dense_until": eval_dense_until, "stable_window": stable_window,
        "rewards": rewards or "default",
    }
    return Q, log, summary


def save_run(Q, log, summary, out_dir: Path = RESULTS_DIR, tag: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tabular_{summary['algo']}_{summary['board']}_seed{summary['seed']}{tag}"
    log.to_csv(out_dir / f"{stem}_log.csv", index=False)
    np.save(out_dir / f"{stem}_Q.npy", Q)
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return stem


# --------------------------------------------------------------------------- command line
def main():
    p = argparse.ArgumentParser(description="Train tabular Q-Learning / SARSA on the Smart Dice board game")
    p.add_argument("--board", default="N30", help="N30, N50, N100 or all")
    p.add_argument("--algo", default="both", help="qlearning, sarsa or both")
    p.add_argument("--episodes", type=int, required=True, help="training episodes (run a small pilot first)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER RUN; saves what it has")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--dense-until", type=int, default=50, help="evaluate after EVERY episode up to this episode")
    p.add_argument("--stable-window", type=int, default=50, help="criterion must hold this many episodes")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.7)
    p.add_argument("--tag", default="", help="suffix for output files (e.g. _pilot)")
    args = p.parse_args()

    boards = list(BOARDS) if args.board == "all" else [args.board]
    algos = list(ALGOS) if args.algo == "both" else [args.algo]
    for b in boards:
        for algo in algos:
            for seed in args.seeds:
                Q, log, summ = train(b, algo, args.episodes, seed, alpha=args.alpha, gamma=args.gamma,
                                     eps_end=args.eps_end, eps_decay_frac=args.eps_decay_frac,
                                     eval_every=args.eval_every, eval_games=args.eval_games,
                                     eval_dense_until=args.dense_until, stable_window=args.stable_window,
                                     time_limit=args.time_limit)
                stem = save_run(Q, log, summ, tag=args.tag)
                status = (f"converged at episode {summ['convergence_episode']}" if summ["converged"]
                          else "NOT converged")
                status += f" | exactly-BFS-optimal from episode {summ['optimal_episode']}"
                if summ["stopped_by_time_limit"]:
                    status += " (stopped by time limit)"
                print(f"==> {stem}: {status}; ran {summ['episodes_run']}/{summ['episodes_planned']} "
                      f"episodes in {summ['elapsed_s']}s\n", flush=True)


if __name__ == "__main__":
    main()