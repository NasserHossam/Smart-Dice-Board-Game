"""
demo.py - play ONE game of the Smart Dice Board Game with a saved policy and print it step by step.

Examples (PowerShell, from the project folder):
  python demos/demo.py --board N30 --source bfs                       # no files needed: BFS-optimal player
  python demos/demo.py --board N30 --source random --seed 3           # random baseline
  python demos/demo.py --board N100 --source tabular --algo sarsa --seed 0
  python demos/demo.py --board N50 --source file --file results\\deep_rl\\SOME_FILE.npy
  python demos/demo.py --list                                         # show candidate policy files in results\\tabular and results\\deep_rl

Policy files accepted (checked by shape, nothing is guessed silently):
  * 2-D array of shape (N, 6)       -> treated as Q-values, greedy action = argmax
  * 1-D array of length N, ints 0-5 -> treated as one action per square (action a = roll a+1)
The environment is deterministic and the greedy policy is deterministic, so one game is the whole
evaluation (the same game every time). The final line compares the result with the BFS optimum.
"""
import argparse
import glob
import os
import time

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # project modules live in src/

from paths import DEEP, TABULAR
from env import BOARDS, SmartDiceEnv, bfs_min_rolls, board_to_text, optimal_roll_sets


def load_policy(path, n):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".csv":
        import pandas as pd
        arr = pd.read_csv(path).to_numpy()
    else:
        raise ValueError(f"unsupported file type: {ext} (use .npy or .csv)")
    arr = np.asarray(arr)
    if arr.shape == (n, 6):
        return arr.argmax(axis=1).astype(int), "Q-table, greedy = argmax (ties -> lowest roll)"
    if arr.ndim == 1 and arr.shape[0] == n and np.all((arr >= 0) & (arr <= 5)) \
            and np.all(arr == np.round(arr)):
        return arr.astype(int), "one action (0-5) per square"
    raise ValueError(f"cannot interpret array of shape {arr.shape} for a board with N={n}; "
                     f"expected ({n}, 6) or ({n},)")


def bfs_policy(board):
    sets = optimal_roll_sets(board)
    pol = np.zeros(board.n, dtype=int)
    for s, rolls in sets.items():
        pol[s - 1] = min(rolls) - 1
    return pol


def find_tabular_file(board, algo, seed):
    pattern = str(TABULAR / f"tabular_{algo or '*'}_{board}_seed{seed if seed is not None else '*'}_Q.npy")
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def list_candidates():
    files = sorted(glob.glob(str(TABULAR / "*.npy")) + glob.glob(str(DEEP / "*policy*")))
    if not files:
        print("No .npy or *policy* files found in results\\tabular or results\\deep_rl.")
    for f in files:
        print(f)


def play(board_name, policy, delay, show_board, rng=None):
    env = SmartDiceEnv(board_name)
    board = env.board
    optimal = optimal_roll_sets(board)
    obs, _ = env.reset()
    path, total, on_optimal, done, term = [1], 0.0, 0, False, False
    print(f"\nBoard {board_name} (N={board.n}), BFS minimum = {bfs_min_rolls(board)} rolls")
    print(board_to_text(board, agent_square=1))
    print()
    while not done:
        sq = env.square
        action = int(rng.integers(0, 6)) if rng is not None else int(policy[obs])
        roll = action + 1
        if roll in optimal.get(sq, set()):
            on_optimal += 1
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        path.append(env.square)
        done = term or trunc
        print(f"roll {env.steps:>3}: square {sq:>3} + {roll} -> {info['square']:>3}  "
              f"{info['event']:<8} reward {reward:>6.1f}  return {total:>7.1f}")
        if show_board:
            print(board_to_text(board, agent_square=env.square, path=path))
            print()
        if delay > 0:
            time.sleep(delay)
    print()
    print("RESULT:", "reached the goal" if term else f"FAILED (cut after {env.max_steps} rolls)")
    print(f"rolls used = {env.steps} | BFS minimum = {bfs_min_rolls(board)} | return = {total:.1f}")
    print(f"rolls that were on a shortest path: {on_optimal}/{env.steps}")
    print("final path:", " -> ".join(str(s) for s in path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", default="N30", choices=list(BOARDS))
    p.add_argument("--source", default="bfs", choices=["bfs", "random", "tabular", "file"])
    p.add_argument("--file", default=None, help="policy file for --source file")
    p.add_argument("--algo", default=None, help="qlearning or sarsa, for --source tabular")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.0, help="seconds to pause after each roll")
    p.add_argument("--show-board", action="store_true", help="print the board after every roll")
    p.add_argument("--list", action="store_true", help="list candidate policy files and exit")
    a = p.parse_args()

    if a.list:
        list_candidates()
        return

    board = BOARDS[a.board]
    rng, policy = None, None
    if a.source == "random":
        rng = np.random.default_rng(a.seed)
        print(f"Policy: uniformly random dice choice (seed {a.seed})")
    elif a.source == "bfs":
        policy = bfs_policy(board)
        print("Policy: BFS-optimal rolls (reference player, not a learned policy)")
    else:
        path = a.file if a.source == "file" else find_tabular_file(a.board, a.algo, a.seed)
        if not path or not os.path.exists(path):
            print("Policy file not found. Candidates:")
            list_candidates()
            print("Then run again with --source file --file <path>.")
            return
        policy, how = load_policy(path, board.n)
        print(f"Policy: {path}\n        interpreted as {how}")
    play(a.board, policy, a.delay, a.show_board, rng)


if __name__ == "__main__":
    main()
