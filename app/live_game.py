"""
live_game.py - watch any player of the Smart Dice Board Game live in the browser.

Start it from the project folder:
    streamlit run app/live_game.py

Pick a board (N30, N50, N100, or the larger Challenge 5 boards N200 and N400) and a player:
  * BFS          - the shortest path (a ruler, not a learned policy)
  * Random       - uniformly random rolls (the "before training" baseline)
  * Q-Learning, SARSA, DQN, PPO, A2C - the trained greedy policies saved in results/ (any seed)
The page only READS saved policies; it never trains or writes anything.
The game itself is the real environment from src/env.py, stepped one roll at a time.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "challenges", "demos"):
    sys.path.insert(0, str(ROOT / sub))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import challenge5_dqn_vs_tabular  # noqa: E402,F401  (registers the N200 and N400 boards)
from demo import bfs_policy, load_policy  # noqa: E402
from env import BOARDS, SmartDiceEnv, bfs_min_rolls, optimal_roll_sets  # noqa: E402
from evaluate import optimal_return  # noqa: E402
from paths import CHALLENGES, DEEP, TABULAR  # noqa: E402

BOARD_NAMES = ["N30", "N50", "N100", "N200", "N400"]
LEARNED = {"Q-Learning": "qlearning", "SARSA": "sarsa", "DQN": "dqn", "PPO": "ppo", "A2C": "a2c"}
EVENT_ICON = {"ladder": "ladder ↑", "snake": "snake ↓", "normal": "normal", "invalid": "invalid (stay)", "goal": "goal 🏁"}


# --------------------------------------------------------------------------- saved policies
@st.cache_data
def policy_files(board: str) -> dict:
    """{player name: {seed label: file}} for every saved policy of this board."""
    out = {}
    ch5 = CHALLENGES / "ch5"
    for name, algo in LEARNED.items():
        files = {}
        if algo in ("qlearning", "sarsa"):
            for f in sorted(TABULAR.glob(f"tabular_{algo}_{board}_seed*_Q.npy")):
                if "pilot" not in f.name:
                    files[f.stem.split("_seed")[1].replace("_Q", "").replace("_extra", "")] = f
            for f in sorted(ch5.glob(f"ch5_{algo}_{board}_seed*_Q.npy")):
                files[f.stem.split("_seed")[1].replace("_Q", "")] = f
        else:
            for f in sorted(DEEP.glob(f"sb3_{algo}_{board}_seed*_policy.npy")):
                files[f.stem.split("_seed")[1].replace("_policy", "")] = f
            for f in sorted(ch5.glob(f"ch5_sb3_{algo}_{board}_seed*_policy.npy")):
                files[f.stem.split("_seed")[1].replace("_policy", "")] = f
        if files:
            out[name] = dict(sorted(files.items(), key=lambda kv: int(kv[0])))
    return out


@st.cache_data
def board_facts(board: str) -> dict:
    b = BOARDS[board]
    return {"bfs": bfs_min_rolls(b), "best_return": optimal_return(b), "optimal": optimal_roll_sets(b)}


# --------------------------------------------------------------------------- board drawing (SVG)
def board_cols(n: int) -> int:
    return 6 if n <= 30 else 10 if n <= 100 else 20


def cell_xy(square: int, n: int, cols: int, cs: float):
    idx = square - 1
    row, k = divmod(idx, cols)
    col = k if row % 2 == 0 else cols - 1 - k
    rows = (n + cols - 1) // cols
    return col * cs, (rows - 1 - row) * cs


def board_svg(board, square: int, path: list, last: tuple | None) -> str:
    n = board.n
    cols = board_cols(n)
    rows = (n + cols - 1) // cols
    cs = 62 if n <= 100 else 34
    w, h = cols * cs, rows * cs
    visited = set(path)
    fs = 13 if n <= 100 else 9
    parts = [f'<svg viewBox="-4 -4 {w + 8} {h + 8}" width="100%" style="max-width:{w + 8}px" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif">',
             '<defs><marker id="ah-g" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1a7f37"/></marker>'
             '<marker id="ah-r" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c62828"/></marker></defs>',
             f'<rect x="-4" y="-4" width="{w + 8}" height="{h + 8}" rx="8" fill="#ffffff"/>']
    for s in range(1, n + 1):
        x, y = cell_xy(s, n, cols, cs)
        fill = "#f6f8fa" if ((s - 1) // cols + (s - 1) % cols) % 2 == 0 else "#ffffff"
        if s in visited:
            fill = "#dbeafe"
        if s == n:
            fill = "#fde68a"
        if s in board.ladders:
            fill = "#dcfce7" if s not in visited else "#bbf7d0"
        if s in board.snakes:
            fill = "#fee2e2" if s not in visited else "#fecaca"
        parts.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" fill="{fill}" stroke="#d0d7de" stroke-width="1"/>')
        parts.append(f'<text x="{x + 3}" y="{y + fs + 1}" font-size="{fs}" fill="#57606a">{s}</text>')
    c = cs / 2
    for s, e in board.ladders.items():
        (x1, y1), (x2, y2) = cell_xy(s, n, cols, cs), cell_xy(e, n, cols, cs)
        parts.append(f'<line x1="{x1 + c}" y1="{y1 + c}" x2="{x2 + c}" y2="{y2 + c}" stroke="#1a7f37" '
                     f'stroke-width="{max(2.5, cs / 14)}" stroke-linecap="round" marker-end="url(#ah-g)" opacity="0.85"/>')
    for s, e in board.snakes.items():
        (x1, y1), (x2, y2) = cell_xy(s, n, cols, cs), cell_xy(e, n, cols, cs)
        mx, my = (x1 + x2) / 2 + c, (y1 + y2) / 2 + c
        dx, dy = (y2 - y1), (x1 - x2)
        ln = max(1.0, (dx * dx + dy * dy) ** 0.5)
        bend = cs * 0.9
        qx, qy = mx + dx / ln * bend, my + dy / ln * bend
        parts.append(f'<path d="M{x1 + c},{y1 + c} Q{qx},{qy} {x2 + c},{y2 + c}" fill="none" stroke="#c62828" '
                     f'stroke-width="{max(2.5, cs / 14)}" stroke-linecap="round" marker-end="url(#ah-r)" opacity="0.85"/>')
    if last is not None:                       # the dice landing square of the last roll
        tx, ty = cell_xy(last[1], n, cols, cs) if last[1] <= n else (None, None)
        if tx is not None:
            parts.append(f'<rect x="{tx + 2}" y="{ty + 2}" width="{cs - 4}" height="{cs - 4}" fill="none" '
                         f'stroke="#f59e0b" stroke-width="3" stroke-dasharray="5 3" rx="4"/>')
    ax, ay = cell_xy(square, n, cols, cs)
    parts.append(f'<circle cx="{ax + c}" cy="{ay + c + fs * 0.3}" r="{cs * 0.28}" fill="#2563eb" stroke="#ffffff" stroke-width="3"/>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- game state
def new_game(board: str, player: str, seed_label: str | None, random_seed: int):
    env = SmartDiceEnv(board)
    env.reset()
    if player == "BFS (shortest path)":
        policy, desc = bfs_policy(env.board), "BFS shortest path (lowest roll among the optimal ones)"
    elif player == "Random":
        policy, desc = None, f"uniformly random rolls (seed {random_seed})"
    else:
        f = policy_files(board)[player][seed_label]
        policy, desc = load_policy(str(f), env.n)
        desc = f"{player} seed {seed_label}: {desc} ({f.relative_to(ROOT).as_posix()})"
    st.session_state.game = {
        "env": env, "policy": policy, "rng": np.random.default_rng(random_seed), "desc": desc,
        "path": [1], "log": [], "total": 0.0, "done": False, "term": False, "on_opt": 0, "last": None,
    }


def step_once():
    g = st.session_state.game
    if g["done"]:
        return
    env = g["env"]
    sq = env.square
    action = int(g["rng"].integers(0, 6)) if g["policy"] is None else int(g["policy"][sq - 1])
    roll = action + 1
    opt = roll in board_facts(env.board.name)["optimal"].get(sq, set())
    _, reward, term, trunc, info = env.step(action)
    g["total"] += reward
    g["on_opt"] += int(opt)
    g["path"].append(env.square)
    g["last"] = (sq, sq + roll)
    g["log"].append({"roll #": env.steps, "from": sq, "dice": roll, "lands on": sq + roll if sq + roll <= env.n else "past N",
                     "event": EVENT_ICON[info["event"]], "to": env.square, "reward": reward,
                     "return": g["total"], "on a shortest path": "yes" if opt else "no"})
    g["done"], g["term"] = bool(term or trunc), bool(term)


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="Smart Dice: live game", page_icon="🎲", layout="wide")
st.title("🎲 Smart Dice Board Game: live")
st.caption("The real environment from src/env.py, played one roll at a time by the player you choose. "
           "Saved policies are only read, never changed.")

with st.sidebar:
    st.header("Game setup")
    board = st.selectbox("Board", BOARD_NAMES, help="N200 and N400 are the larger boards from Challenge 5.")
    files = policy_files(board)
    players = ["BFS (shortest path)", "Random"] + list(files)
    player = st.selectbox("Player", players, index=min(2, len(players) - 1))
    seed_label, random_seed = None, 0
    if player in files:
        seed_label = st.selectbox("Trained seed", list(files[player]))
    if player == "Random":
        random_seed = st.number_input("Random seed", 0, 10_000, 3)
    speed = st.slider("Seconds per roll (Play)", 0.05, 2.0, 0.5, 0.05)
    st.divider()
    b = BOARDS[board]
    facts = board_facts(board)
    st.markdown(f"**{board}**: {b.n} squares, {len(b.ladders)} ladders, {len(b.snakes)} snakes  \n"
                f"BFS minimum: **{facts['bfs']} rolls**  \nBest possible return: **{facts['best_return']:.0f}**")
    st.markdown("🟩 ladder start · 🟥 snake head · 🟨 goal · 🟦 squares visited  \n"
                "Green arrow = ladder, red curve = snake, dashed box = where the dice landed.")

config = (board, player, seed_label, int(random_seed))
if st.session_state.get("config") != config or "game" not in st.session_state:
    st.session_state.config = config
    new_game(board, player, seed_label, int(random_seed))

c1, c2, c3, _ = st.columns([1, 1, 1, 3])
play = c1.button("▶ Play", type="primary", use_container_width=True)
if c2.button("⏭ Step", use_container_width=True):
    step_once()
if c3.button("↺ Reset", use_container_width=True):
    new_game(board, player, seed_label, int(random_seed))

left, right = st.columns([3, 2])
board_box = left.empty()
with right:
    status_box = st.empty()
    metrics_box = st.empty()
    log_box = st.empty()


def render():
    g = st.session_state.game
    env = g["env"]
    board_box.markdown(board_svg(env.board, env.square, g["path"], g["last"]), unsafe_allow_html=True)
    if g["done"]:
        if g["term"]:
            gap = env.steps - facts["bfs"]
            status_box.success(f"Reached the goal in **{env.steps} rolls** (BFS minimum {facts['bfs']}"
                               f"{', optimal' if gap == 0 else f', {gap} more'}) with return **{g['total']:.0f}** "
                               f"(best possible {facts['best_return']:.0f}).")
        else:
            status_box.error(f"Did not finish: the game was cut after {env.max_steps} rolls (5 × N).")
    else:
        status_box.info(f"Player: {g['desc']}")
    with metrics_box.container():
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Square", f"{env.square} / {env.n}")
        m2.metric("Rolls", env.steps, help=f"BFS minimum = {facts['bfs']}")
        m3.metric("Return", f"{g['total']:.0f}")
        m4.metric("On a shortest path", f"{g['on_opt']}/{env.steps}" if env.steps else "–")
    if g["log"]:
        log_box.dataframe(list(reversed(g["log"])), hide_index=True, use_container_width=True, height=320)
    else:
        log_box.caption("Press ▶ Play to watch the whole game, or ⏭ Step for one roll at a time.")


render()
if play:
    while not st.session_state.game["done"]:
        step_once()
        render()
        time.sleep(speed)
