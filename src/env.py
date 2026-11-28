"""
env.py - Smart Dice Board Game as a Gymnasium environment.

RULES (all documented here so the report can quote them)
--------------------------------------------------------
* Board has N squares, numbered 1..N. The agent starts on square 1; the goal is square N.
* STATE  : the current square. Observation = Discrete(N), obs = square - 1 (0-indexed).
* ACTION : Discrete(6). action a means "roll the dice value a + 1" (the agent chooses the roll).
* TRANSITION (deterministic):
    target = square + roll
    - if target > N                : INVALID MOVE (see rule below)
    - else if target holds a ladder: agent moves up to the ladder top immediately
    - else if target holds a snake : agent moves down to the snake tail immediately
    - else                         : agent moves to target
  Jumps are applied once and never chain (validate_board forbids a jump ending on a jump start).
* INVALID-MOVE RULE (one consistent rule): PENALIZE AND STAY.
  The agent stays on the same square, receives the invalid reward, and the move still
  counts as one roll/step.
* REWARDS (defaults; any of them can be overridden to test reward design):
    reach goal +100 | land on ladder +20 | normal square -1 | land on snake -20 | invalid -10
  The reward depends on the square the dice lands on (the ladder base / snake head).
* TERMINATION: reaching square N (terminated=True).
* TRUNCATION : an episode is cut after max_steps rolls (default 5 * N) so that a bad
  policy cannot loop forever. A truncated episode counts as NOT successful.
* RANDOMNESS: the environment itself is deterministic. Seeds only matter for the agents
  (exploration, network init). Board layouts below are fixed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEFAULT_REWARDS = {"goal": 100.0, "ladder": 20.0, "normal": -1.0, "snake": -20.0, "invalid": -10.0}


# --------------------------------------------------------------------------- boards
@dataclass
class Board:
    name: str
    n: int
    ladders: dict  # {start_square: end_square}, end > start
    snakes: dict   # {head_square: tail_square}, tail < head
    jump_map: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.jump_map = {**self.ladders, **self.snakes}


# Fixed layouts (documented for the report). Do not edit after you start running experiments.
BOARDS = {
    "N30": Board("N30", 30,
                 ladders={5: 13, 11: 19, 20: 26},
                 snakes={14: 4, 23: 9, 27: 15}),
    "N50": Board("N50", 50,
                 ladders={4: 14, 13: 24, 22: 33, 30: 41, 38: 46},
                 snakes={19: 7, 27: 11, 35: 21, 43: 29, 48: 32}),
    "N100": Board("N100", 100,
                  ladders={6: 17, 15: 27, 25: 38, 41: 52, 55: 66, 64: 77, 79: 90, 86: 94},
                  snakes={18: 8, 34: 19, 45: 29, 58: 40, 69: 50, 74: 56, 91: 72, 97: 81}),
}


def validate_board(b: Board) -> None:
    """Raise ValueError if the layout breaks any rule."""
    errs = []
    n = b.n
    for s, e in b.ladders.items():
        if not e > s:
            errs.append(f"ladder {s}->{e} does not go up")
    for s, e in b.snakes.items():
        if not e < s:
            errs.append(f"snake {s}->{e} does not go down")
    starts = list(b.ladders) + list(b.snakes)
    if len(set(starts)) != len(starts):
        errs.append("a square holds two jumps")
    for s in starts:
        if not 1 < s < n:
            errs.append(f"jump starts on square {s} (not allowed on 1 or N)")
    for e in b.jump_map.values():
        if not 1 <= e <= n:
            errs.append(f"jump ends outside the board: {e}")
        if e in b.jump_map:
            errs.append(f"jump ends on another jump start ({e}); chaining not allowed")
    if errs:
        raise ValueError(f"Invalid board {b.name}: " + "; ".join(errs))
    if bfs_min_rolls(b) is None:
        raise ValueError(f"Board {b.name}: goal is unreachable")


# --------------------------------------------------------------------------- BFS optimum
def next_square(board: Board, square: int, roll: int):
    """Square after rolling `roll` from `square`, or None if the move is invalid."""
    target = square + roll
    if target > board.n:
        return None
    return board.jump_map.get(target, target)


def bfs_distances_to_goal(board: Board) -> np.ndarray:
    """dist[s] = minimum number of rolls from square s to the goal (-1 if unreachable).
    Index 0 is unused. Computed by BFS on the reversed transition graph."""
    n = board.n
    reverse = {s: [] for s in range(1, n + 1)}
    for s in range(1, n):
        for roll in range(1, 7):
            nxt = next_square(board, s, roll)
            if nxt is not None:
                reverse[nxt].append(s)
    dist = np.full(n + 1, -1, dtype=int)
    dist[n] = 0
    q = deque([n])
    while q:
        cur = q.popleft()
        for prev in reverse[cur]:
            if dist[prev] == -1:
                dist[prev] = dist[cur] + 1
                q.append(prev)
    return dist


def bfs_min_rolls(board: Board, start: int = 1):
    """True minimum number of rolls from `start` to the goal (None if unreachable)."""
    d = bfs_distances_to_goal(board)[start]
    return None if d < 0 else int(d)


def optimal_roll_sets(board: Board) -> dict:
    """For every square, the set of rolls (1..6) that lie on a shortest path to the goal.
    Used later to measure how often the learned policy picks an optimal roll."""
    dist = bfs_distances_to_goal(board)
    out = {}
    for s in range(1, board.n):
        good = set()
        for roll in range(1, 7):
            nxt = next_square(board, s, roll)
            if nxt is not None and dist[nxt] >= 0 and dist[nxt] + 1 == dist[s]:
                good.add(roll)
        out[s] = good
    return out


# --------------------------------------------------------------------------- text rendering
def board_to_text(board: Board, agent_square: int | None = None, path: list | None = None) -> str:
    """ASCII board, 10 squares per row, top row = highest squares (zig-zag numbering).
    L = ladder base, S = snake head, * = agent, . = square visited on `path`."""
    width = 10
    rows = (board.n + width - 1) // width
    visited = set(path or [])
    lines = []
    for r in range(rows - 1, -1, -1):
        cells = list(range(r * width + 1, min((r + 1) * width, board.n) + 1))
        if r % 2 == 1:
            cells = cells[::-1]
        out = []
        for sq in cells:
            mark = " "
            if sq in board.ladders:
                mark = "L"
            elif sq in board.snakes:
                mark = "S"
            elif sq in visited:
                mark = "."
            if sq == agent_square:
                mark = "*"
            out.append(f"{sq:>3}{mark}")
        lines.append(" ".join(out))
    legend = [f"Ladders: {dict(sorted(board.ladders.items()))}",
              f"Snakes : {dict(sorted(board.snakes.items()))}"]
    return "\n".join(lines + legend)


# --------------------------------------------------------------------------- environment
class SmartDiceEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, board: Board | str = "N30", rewards: dict | None = None,
                 max_steps: int | None = None, render_mode: str | None = None):
        super().__init__()
        self.board = BOARDS[board] if isinstance(board, str) else board
        validate_board(self.board)
        self.n = self.board.n
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.max_steps = int(max_steps) if max_steps is not None else 5 * self.n
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(self.n)   # obs = square - 1
        self.action_space = spaces.Discrete(6)             # action = roll - 1
        self.square = 1
        self.steps = 0

    def _obs(self) -> int:
        return int(self.square - 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.square = 1
        self.steps = 0
        return self._obs(), {"square": self.square}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in 0..5, got {action}")
        roll = action + 1
        self.steps += 1
        target = self.square + roll
        b = self.board
        if target > self.n:
            reward, event = self.rewards["invalid"], "invalid"      # stay where we are
        else:
            final = b.jump_map.get(target, target)
            if final == self.n:
                reward, event = self.rewards["goal"], "goal"
            elif target in b.ladders:
                reward, event = self.rewards["ladder"], "ladder"
            elif target in b.snakes:
                reward, event = self.rewards["snake"], "snake"
            else:
                reward, event = self.rewards["normal"], "normal"
            self.square = final
        terminated = self.square == self.n
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"square": self.square, "roll": roll, "event": event}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        return board_to_text(self.board, agent_square=self.square)


# --------------------------------------------------------------------------- self-test
def run_random_policy(board_name: str, episodes: int = 300, seed: int = 0):
    """Baseline: uniformly random dice choice. Returns (success %, mean rolls, mean return)."""
    rng = np.random.default_rng(seed)
    env = SmartDiceEnv(board_name)
    wins, moves, rets = 0, [], []
    for _ in range(episodes):
        env.reset()
        done, ret = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(int(rng.integers(0, 6)))
            ret += r
            done = term or trunc
        wins += int(term)
        moves.append(env.steps)
        rets.append(ret)
    return 100.0 * wins / episodes, float(np.mean(moves)), float(np.mean(rets))


def self_test():
    # 1) fixed layouts are valid
    for name, b in BOARDS.items():
        validate_board(b)
        print(f"[ok] layout {name}: N={b.n}, {len(b.ladders)} ladders, {len(b.snakes)} snakes")

    # 2) hand-checked mini board: N=10, ladder 3->8, snake 6->2
    mini = Board("mini", 10, ladders={3: 8}, snakes={6: 2})
    validate_board(mini)
    assert bfs_min_rolls(mini) == 2, "BFS on mini board should be 2 (roll 2 -> ladder to 8, roll 2 -> goal)"
    env = SmartDiceEnv(mini)
    obs, _ = env.reset()
    assert obs == 0
    obs, r, term, trunc, info = env.step(1)            # roll 2 -> square 3 -> ladder to 8
    assert (obs, r, term, info["event"]) == (7, 20.0, False, "ladder")
    obs, r, term, trunc, info = env.step(2)            # roll 3 -> 11 > 10 -> invalid, stay on 8
    assert (obs, r, term, info["event"]) == (7, -10.0, False, "invalid")
    obs, r, term, trunc, info = env.step(1)            # roll 2 -> square 10 = goal
    assert (obs, r, term, info["event"]) == (9, 100.0, True, "goal")
    env.reset()
    obs, r, term, trunc, info = env.step(4)            # roll 5 -> square 6 -> snake to 2
    assert (obs, r, info["event"]) == (1, -20.0, "snake")
    obs, r, term, trunc, info = env.step(0)            # roll 1 -> square 3 -> ladder? no: 2+1=3 -> ladder to 8
    assert info["event"] == "ladder" and obs == 7
    print("[ok] transitions and rewards on mini board (ladder, snake, invalid, goal)")

    # 3) truncation
    env = SmartDiceEnv(mini, max_steps=3)
    env.reset()
    for _ in range(3):
        _, _, term, trunc, _ = env.step(5)             # roll 6 from 1 -> 7, then 13 invalid, ...
    assert trunc and not term
    print("[ok] truncation after max_steps")

    # 4) Gymnasium's own checker
    from gymnasium.utils.env_checker import check_env
    for name in BOARDS:
        check_env(SmartDiceEnv(name), skip_render_check=True)
    print("[ok] gymnasium check_env passed on all three boards")

    # 5) every state has at least one optimal roll
    for name, b in BOARDS.items():
        sets = optimal_roll_sets(b)
        assert all(len(v) >= 1 for v in sets.values())
    print("[ok] optimal roll sets non-empty for every square")

    # 6) BFS optimum + random baseline per board
    print("\nBoard   N    BFS min rolls   random: success%  mean rolls  mean return")
    for name, b in BOARDS.items():
        succ, mv, ret = run_random_policy(name)
        print(f"{name:<7}{b.n:<5}{bfs_min_rolls(b):<16}{succ:>14.1f}{mv:>12.1f}{ret:>13.1f}")

    print("\nBoard N30 (L = ladder base, S = snake head):")
    print(board_to_text(BOARDS["N30"]))
    print("\nSELF-TEST PASSED")


if __name__ == "__main__":
    self_test()
