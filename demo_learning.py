"""
demo_learning.py - WATCH THE AGENT LEARN (Q-Learning and SARSA, same board, same seed).

What it prints
  1. The board.
  2. BEFORE training: one game with a random policy.
  3. For each algorithm: after chosen training episodes (snapshots), the greedy policy plays one game.
     You see rolls, return, success and the path improve as episodes go by. Nothing is
     pre-computed: the agent only ever sees the environment's rewards. BFS is used only at the
     very end as a ruler to measure the result.
  4. AFTER training: the final greedy game of each algorithm, step by step, on the board.
  5. Q-Learning vs SARSA side by side.
  6. A self-check: the training loop here must give exactly the same Q-table as tabular.train()
     (same seed, same settings). If it does not, the script says so.

Note: the greedy policy is deterministic and the environment is deterministic, so one game per
snapshot is the whole evaluation. Ties in Q-values are broken toward the lowest roll (argmax), so
an untrained table (all zeros) always picks roll 1.

Usage (from the project folder):
  python demo_learning.py                                   # N30, seed 0, 300 episodes
  python demo_learning.py --board N100 --episodes 500 --snapshots 1 5 10 20 50 100 500
  python demo_learning.py --board N50 --seed 1 --quiet-final    # skip the step-by-step final games
"""
import argparse

import numpy as np

from env import BOARDS, SmartDiceEnv, bfs_min_rolls, board_to_text
from tabular import ALGOS, epsilon_greedy, train

NAMES = {"qlearning": "Q-Learning", "sarsa": "SARSA"}


def play_greedy(board_name, Q, verbose=False):
    """One greedy game (argmax, ties -> lowest roll). Returns dict with success, rolls, ret, path."""
    env = SmartDiceEnv(board_name)
    obs, _ = env.reset()
    path, ret, done, term, lines = [1], 0.0, False, False, []
    while not done:
        sq = env.square
        a = int(Q[obs].argmax())
        obs, r, term, trunc, info = env.step(a)
        ret += r
        path.append(env.square)
        lines.append(f"  roll {env.steps:>3}: square {sq:>3} + {a + 1} -> {info['square']:>3}  "
                     f"{info['event']:<8} reward {r:>6.1f}  return {ret:>7.1f}")
        done = term or trunc
    return {"success": bool(term), "rolls": env.steps, "ret": ret, "path": path, "lines": lines}


def play_random(board_name, seed):
    rng = np.random.default_rng(seed)
    env = SmartDiceEnv(board_name)
    env.reset()
    path, ret, done, term = [1], 0.0, False, False
    while not done:
        _, r, term, trunc, _ = env.step(int(rng.integers(0, 6)))
        ret += r
        path.append(env.square)
        done = term or trunc
    return {"success": bool(term), "rolls": env.steps, "ret": ret, "path": path}


def short_path(path, limit=14):
    if len(path) <= limit:
        return " -> ".join(map(str, path))
    return " -> ".join(map(str, path[:limit])) + f" -> ... ({len(path) - 1} rolls in total)"


def learn_with_snapshots(board_name, algo, episodes, seed, snapshots,
                         alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.7):
    """Same training loop as tabular.train(), but it keeps a copy of Q at the snapshot episodes."""
    rng = np.random.default_rng(seed)
    env = SmartDiceEnv(board_name)
    Q = np.zeros((env.n, 6))
    decay = (eps_end / eps_start) ** (1.0 / max(1, int(eps_decay_frac * episodes)))
    eps = eps_start
    snaps = {}
    for ep in range(1, episodes + 1):
        eps_used = eps
        obs, _ = env.reset(seed=seed if ep == 1 else None)
        a = epsilon_greedy(Q[obs], eps, rng)
        done = False
        while not done:
            nobs, r, term, trunc, _ = env.step(a)
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
        eps = max(eps_end, eps * decay)
        if ep in snapshots:
            snaps[ep] = (Q.copy(), eps_used)
    return Q, snaps


def main():
    p = argparse.ArgumentParser(description="Watch Q-Learning and SARSA learn the Smart Dice board game")
    p.add_argument("--board", default="N30", choices=list(BOARDS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--snapshots", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 300])
    p.add_argument("--quiet-final", action="store_true", help="do not print the final games step by step")
    a = p.parse_args()

    snapshots = sorted({s for s in a.snapshots if 1 <= s <= a.episodes} | {a.episodes})
    board = BOARDS[a.board]
    bfs = bfs_min_rolls(board)
    print(f"=== {a.board} (N={board.n}), seed {a.seed}, {a.episodes} training episodes; "
          f"BFS minimum = {bfs} rolls ===\n")
    print(board_to_text(board, agent_square=1))

    rnd = play_random(a.board, a.seed)
    print(f"\n--- BEFORE training: random policy (one game) ---")
    print(f"{'reached goal' if rnd['success'] else 'FAILED'} in {rnd['rolls']} rolls, return {rnd['ret']:.1f}")
    print("path:", short_path(rnd["path"]))

    results = {}
    for algo in ALGOS:
        Q, snaps = learn_with_snapshots(a.board, algo, a.episodes, a.seed, set(snapshots))
        results[algo] = (Q, snaps)
        print(f"\n--- {NAMES[algo]}: greedy game after N training episodes ---")
        print(f"{'episodes':>8} {'epsilon':>8} {'goal?':>6} {'rolls':>6} {'return':>8}   path")
        for ep in snapshots:
            Qs, eps_used = snaps[ep]
            g = play_greedy(a.board, Qs)
            print(f"{ep:>8} {eps_used:>8.3f} {'yes' if g['success'] else 'NO':>6} {g['rolls']:>6} "
                  f"{g['ret']:>8.1f}   {short_path(g['path'])}")

    if not a.quiet_final:
        for algo in ALGOS:
            Q, _ = results[algo]
            g = play_greedy(a.board, Q)
            print(f"\n--- AFTER training: {NAMES[algo]}, final greedy game ---")
            print("\n".join(g["lines"][:40]))
            print(board_to_text(board, path=g["path"]))

    print("\n--- Q-Learning vs SARSA (final greedy game) ---")
    print(f"{'':<12}{'goal?':>7}{'rolls':>7}{'return':>9}   path")
    finals = {}
    for algo in ALGOS:
        g = play_greedy(a.board, results[algo][0])
        finals[algo] = g
        print(f"{NAMES[algo]:<12}{'yes' if g['success'] else 'NO':>7}{g['rolls']:>7}{g['ret']:>9.1f}   "
              f"{short_path(g['path'])}")
    print(f"{'random':<12}{'yes' if rnd['success'] else 'NO':>7}{rnd['rolls']:>7}{rnd['ret']:>9.1f}   (before training)")
    print(f"{'BFS ruler':<12}{'':>7}{bfs:>7}   (minimum possible rolls)")
    same = finals["qlearning"]["path"] == finals["sarsa"]["path"]
    print("The two algorithms ended on the " + ("SAME path." if same else "DIFFERENT paths."))

    print("\n--- self-check against tabular.train() ---")
    for algo in ALGOS:
        Qref, _, _ = train(a.board, algo, a.episodes, a.seed, verbose=False)
        ok = np.array_equal(Qref, results[algo][0])
        print(f"{NAMES[algo]:<11}: " + ("[ok] identical Q-table" if ok else "[MISMATCH] Q-table differs"))


if __name__ == "__main__":
    main()
