"""
experiments.py - two small TABULAR experiments that answer report questions:

  reward  : How does the reward function change the learned policy?
            Same board (default N50), five reward functions, Q-Learning and SARSA, several seeds.
            After training, the greedy policy is replayed under the ORIGINAL rewards and we record
            success, rolls, snakes hit, ladders used. We also check whether the agent reached the
            best possible return for ITS OWN reward (dynamic programming). NaN = that reward allows
            an endless positive loop, so no finite optimum exists.
  layout  : How do the NUMBER and LOCATION of snakes and ladders affect learning?
            Six N=50 variants of the default layout (fixed and documented below).

Both use the same tabular.train() as the main runs, so epsilon-greedy, alpha, gamma are identical.
Convergence for the layout experiment uses the same rule as the main results: from episode E until
the end of training every checkpoint has success >= 95% and rolls <= 1.10 * BFS optimum.

Usage:
  python src/experiments.py --which both --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 2000
  python src/experiments.py --which reward --board N100 --seeds 0 1 2 3 4
Outputs: results/experiments/exp_reward_<board>_runs.csv + _summary.csv, exp_layout_runs.csv + _summary.csv,
         results/figures/plot_exp_reward.png, plot_exp_layout.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_tabular import stable_to_end
from env import BOARDS, Board, SmartDiceEnv, bfs_min_rolls, validate_board
from evaluate import optimal_return
from paths import EXPERIMENTS, FIGURES
from tabular import train

ALGOS = ("qlearning", "sarsa")
NAMES = {"qlearning": "Q-Learning", "sarsa": "SARSA"}

# --------------------------------------------------------------------------- experiment definitions
REWARD_CONFIGS = {
    "default": {},                                                  # goal 100, ladder 20, normal -1, snake -20, invalid -10
    "step_cost_only": {"ladder": -1.0, "snake": -1.0},              # ladders/snakes cost the same as a normal square
    "harsh_snake": {"snake": -100.0},                               # snakes hurt much more
    "no_step_cost": {"normal": 0.0},                                # moving is free
    "big_ladder": {"ladder": 100.0},                                # ladders hugely rewarded
}

_N50 = BOARDS["N50"]
LAYOUTS = {
    "base (5L,5S)": _N50,
    "no_snakes (5L,0S)": Board("no_snakes", 50, ladders=dict(_N50.ladders), snakes={}),
    "no_ladders (0L,5S)": Board("no_ladders", 50, ladders={}, snakes=dict(_N50.snakes)),
    "many_snakes (5L,10S)": Board("many_snakes", 50, ladders=dict(_N50.ladders),
                                  snakes={**_N50.snakes, 9: 2, 16: 6, 25: 12, 36: 20, 45: 34}),
    "snakes_early (5L,5S)": Board("snakes_early", 50, ladders=dict(_N50.ladders),
                                  snakes={9: 3, 11: 2, 16: 6, 18: 5, 20: 7}),
    "snakes_late (5L,5S)": Board("snakes_late", 50, ladders=dict(_N50.ladders),
                                 snakes={40: 25, 43: 26, 44: 31, 47: 34, 49: 36}),
}


# --------------------------------------------------------------------------- helpers
def play_greedy(board: Board, Q: np.ndarray, rewards: dict | None = None):
    """Greedy play (first-index tie break). Returns (success, rolls, return, event counts)."""
    env = SmartDiceEnv(board, rewards=rewards)
    obs, _ = env.reset()
    done, ret, term = False, 0.0, False
    counts = {"ladder": 0, "snake": 0, "invalid": 0}
    while not done:
        obs, r, term, trunc, info = env.step(int(Q[obs].argmax()))
        ret += r
        if info["event"] in counts:
            counts[info["event"]] += 1
        done = term or trunc
    return bool(term), env.steps, ret, counts


def random_rolls(board: Board, games: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    env = SmartDiceEnv(board)
    steps = []
    for _ in range(games):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(0, 6)))
            done = term or trunc
        steps.append(env.steps)
    return float(np.mean(steps))


def _med(v) -> str:
    v = pd.Series(v).dropna().astype(float)
    return "-" if len(v) == 0 else f"{v.median():.0f} [{v.min():.0f}-{v.max():.0f}]"


# --------------------------------------------------------------------------- reward experiment
def run_reward(board_name: str, seeds, episodes: int, time_limit):
    board = BOARDS[board_name]
    bfs = bfs_min_rolls(board)
    rows = []
    for cfg, rw in REWARD_CONFIGS.items():
        own_opt = optimal_return(board, rw)
        for algo in ALGOS:
            t0 = time.time()
            for seed in seeds:
                Q, _, _ = train(board_name, algo, episodes, seed, rewards=rw or None, eval_every=500,
                                eval_dense_until=0, eval_games=1, time_limit=time_limit, verbose=False)
                ok, rolls, ret, ev = play_greedy(board, Q)
                _, _, ret_own, _ = play_greedy(board, Q, rw or None)
                rows.append({"config": cfg, "algo": algo, "seed": seed, "bfs": bfs, "success": int(ok),
                             "rolls": rolls, "orig_return": ret, "snakes_hit": ev["snake"],
                             "ladders_used": ev["ladder"], "invalid_moves": ev["invalid"],
                             "reached_bfs_rolls": int(ok and rolls == bfs),
                             "own_optimum": own_opt, "own_return": ret_own,
                             "reached_own_optimum": (int(abs(ret_own - own_opt) < 1e-6) if not np.isnan(own_opt) else np.nan)})
            print(f"  reward '{cfg}' {NAMES[algo]}: {len(seeds)} seeds in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENTS / f"exp_reward_{board_name}_runs.csv", index=False)
    summ = (df.groupby(["config", "algo"], sort=False)
            .agg(n=("seed", "count"), success_share=("success", "mean"), rolls_median=("rolls", "median"),
                 reached_bfs_rolls=("reached_bfs_rolls", "mean"), orig_return=("orig_return", "mean"),
                 snakes_hit=("snakes_hit", "mean"), ladders_used=("ladders_used", "mean"),
                 own_optimum_value=("own_optimum", "first"), reached_own_optimum=("reached_own_optimum", "mean"))
            .round(2).reset_index())
    summ.to_csv(EXPERIMENTS / f"exp_reward_{board_name}_summary.csv", index=False)
    print(f"\nREWARD EXPERIMENT on {board_name} (BFS optimum = {bfs} rolls; success/rolls/return measured under the "
          f"ORIGINAL rewards; own_optimum_value = best possible return for that reward, NaN = endless loop possible)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cfgs = list(REWARD_CONFIGS)
    x = np.arange(len(cfgs))
    for k, algo in enumerate(ALGOS):
        s = summ[summ.algo == algo].set_index("config").loc[cfgs]
        axes[0].bar(x + (k - 0.5) * 0.38, s["success_share"] * 100, 0.38, label=NAMES[algo])
        axes[1].bar(x + (k - 0.5) * 0.38, s["snakes_hit"], 0.38, label=NAMES[algo])
        axes[2].bar(x + (k - 0.5) * 0.38, s["rolls_median"], 0.38, label=NAMES[algo])
    axes[0].set_title("greedy policy reaches the goal (% of seeds)")
    axes[1].set_title("snakes hit on the greedy path (mean)")
    axes[2].set_title(f"rolls used (median; BFS = {bfs})")
    axes[2].axhline(bfs, color="k", ls="--", lw=1)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=25, ha="right")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.suptitle(f"Effect of the reward function ({board_name}, evaluated under the original rewards)")
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_exp_reward.png", dpi=130)
    plt.close(fig)
    return summ


# --------------------------------------------------------------------------- layout experiment
def run_layout(seeds, episodes: int, time_limit):
    rows = []
    for name, board in LAYOUTS.items():
        validate_board(board)
        bfs = bfs_min_rolls(board)
        rnd = random_rolls(board)
        for algo in ALGOS:
            t0 = time.time()
            for seed in seeds:
                _, log, _ = train(board, algo, episodes, seed, time_limit=time_limit, verbose=False)
                ev = log.dropna(subset=["eval_success"])
                loose = ((ev["eval_success"] >= 0.95) & (ev["eval_moves"] <= 1.10 * bfs)).to_numpy()
                conv = stable_to_end(ev["episode"].to_numpy(), loose)
                rows.append({"layout": name, "ladders": len(board.ladders), "snakes": len(board.snakes),
                             "bfs": bfs, "random_rolls": round(rnd, 1), "algo": algo, "seed": seed,
                             "conv_episode": np.nan if conv is None else conv,
                             "final_moves": float(ev["eval_moves"].iloc[-1]),
                             "final_return": float(ev["eval_return"].iloc[-1])})
            print(f"  layout '{name}' {NAMES[algo]}: {len(seeds)} seeds in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENTS / "exp_layout_runs.csv", index=False)
    out = []
    for (layout, algo), g in df.groupby(["layout", "algo"], sort=False):
        out.append({"layout": layout, "algo": NAMES[algo], "n": len(g), "BFS": int(g["bfs"].iloc[0]),
                    "random_rolls": g["random_rolls"].iloc[0], "conv_episodes median[min-max]": _med(g["conv_episode"]),
                    "not_converged": int(g["conv_episode"].isna().sum()),
                    "final_rolls_mean": round(g["final_moves"].mean(), 2)})
    summ = pd.DataFrame(out)
    summ.to_csv(EXPERIMENTS / "exp_layout_summary.csv", index=False)
    print("\nLAYOUT EXPERIMENT (N=50; random_rolls = mean rolls of a random player; convergence in training episodes)")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = list(LAYOUTS)
    x = np.arange(len(names))
    for k, algo in enumerate(ALGOS):
        med, lo, hi = [], [], []
        for n in names:
            v = df[(df.layout == n) & (df.algo == algo)]["conv_episode"].dropna()
            med.append(v.median() if len(v) else np.nan)
            lo.append(v.min() if len(v) else np.nan)
            hi.append(v.max() if len(v) else np.nan)
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(x + (k - 0.5) * 0.2, med, yerr=[med - lo, hi - med], fmt="o", capsize=3, label=NAMES[algo])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("training episodes to convergence (median, bars = min-max)")
    ax.set_title("Effect of the number and location of snakes and ladders (N=50)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_exp_layout.png", dpi=130)
    plt.close(fig)
    return summ


def main():
    p = argparse.ArgumentParser(description="Reward-design and layout experiments (tabular)")
    p.add_argument("--which", default="both", choices=["reward", "layout", "both"])
    p.add_argument("--board", default="N50", help="board for the reward experiment")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--time-limit", type=float, default=120, help="max seconds per training run")
    a = p.parse_args()
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if a.which in ("reward", "both"):
        run_reward(a.board, a.seeds, a.episodes, a.time_limit)
    if a.which in ("layout", "both"):
        run_layout(a.seeds, a.episodes, a.time_limit)


if __name__ == "__main__":
    main()
