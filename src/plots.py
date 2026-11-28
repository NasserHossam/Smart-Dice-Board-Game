"""
plots.py - all required figures, saved as PNG in results/figures/.

  plot_reward_vs_episodes.png     total (training) reward per episode vs episodes, all 5 algorithms, 3 boards
  plot_moves_vs_episodes.png      moves (rolls) per training episode vs episodes
  plot_success_vs_episodes.png    top: training success rate; bottom: share of seeds whose GREEDY policy succeeds
  plot_q_vs_sarsa.png             Q-Learning vs SARSA: reward curve, greedy rolls curve, convergence episodes
  plot_deep_vs_timesteps.png      DQN / PPO / A2C: greedy rolls vs timesteps and convergence timesteps
  plot_convergence.png            episodes to convergence, every seed, every algorithm
  plot_board_size.png             episodes to convergence as the board grows (N = 30, 50, 100)

How to read them (say this in the report):
  * Curves are the MEAN over seeds (shaded = +-1 sd when there is more than one seed), smoothed with a
    rolling mean, and the x-axis is logarithmic because most learning happens in the first episodes.
  * Training curves include exploration (epsilon-greedy or the stochastic policy). Training success is
    often ~100% from the start, because the board has no dead ends; learning shows up as fewer moves
    and higher reward, and in the greedy-policy panels.

Usage:  python src/plots.py
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_tabular import PAT as TAB_PAT
from env import BOARDS, bfs_min_rolls
from evaluate import ALGO_ORDER, NAMES, build_per_run
from paths import DEEP, FIGURES
from paths import TABULAR as TABULAR_DIR   # TABULAR below is the tuple of tabular algorithm names

COLORS = {"qlearning": "#1f77b4", "sarsa": "#ff7f0e", "dqn": "#2ca02c", "ppo": "#d62728", "a2c": "#9467bd"}
TABULAR = ("qlearning", "sarsa")


# --------------------------------------------------------------------------- loading
def load_runs(algo: str, board: str) -> list:
    """List of dicts: per-episode arrays (ret, moves, succ) and checkpoint table ev."""
    runs = []
    if algo in TABULAR:
        for p in sorted(TABULAR_DIR.glob(f"tabular_{algo}_{board}_seed*_log.csv")):
            if "pilot" in p.name or not TAB_PAT.match(p.name):
                continue
            d = pd.read_csv(p)
            ev = d.dropna(subset=["eval_success"])[["episode", "eval_success", "eval_moves", "eval_return"]].copy()
            ev["timesteps"] = np.nan
            runs.append({"ret": d["train_return"].to_numpy(), "moves": d["train_moves"].to_numpy(),
                         "succ": d["train_success"].to_numpy(float), "ev": ev})
    else:
        for p in sorted(DEEP.glob(f"sb3_{algo}_{board}_seed*_episodes.csv")):
            if "pilot" in p.name:
                continue
            e = pd.read_csv(p)
            ev = pd.read_csv(str(p).replace("_episodes.csv", "_evals.csv")).rename(columns={"episodes": "episode"})
            runs.append({"ret": e["return"].to_numpy(), "moves": e["moves"].to_numpy(),
                         "succ": e["success"].to_numpy(float), "ev": ev})
    return runs


def smooth(a, w):
    return pd.Series(np.asarray(a, dtype=float)).rolling(w, min_periods=1).mean().to_numpy()


def mean_curve(runs, key, window):
    arrs = [smooth(r[key], window) for r in runs]
    L = min(len(a) for a in arrs)
    M = np.vstack([a[:L] for a in arrs])
    return np.arange(1, L + 1), M.mean(0), M.std(0)


def _step_matrix(runs, col, xcol="episode"):
    xmax = min(r["ev"][xcol].max() for r in runs)
    grid = np.unique(np.round(np.logspace(0, np.log10(max(xmax, 2)), 250)).astype(int))
    vals = []
    for r in runs:
        x, y = r["ev"][xcol].to_numpy(), r["ev"][col].to_numpy()
        idx = np.searchsorted(x, grid, side="right") - 1
        vals.append(np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan))
    return grid, np.vstack(vals)


def step_curve(runs, col, xcol="episode"):
    """Mean +- sd over runs of checkpoint column `col`, as a step function of `xcol`."""
    grid, V = _step_matrix(runs, col, xcol)
    return grid, np.nanmean(V, 0), np.nanstd(V, 0)


def step_median(runs, col, xcol="episode"):
    """Median and inter-quartile band over runs (robust to a few failed 500-roll policies)."""
    grid, V = _step_matrix(runs, col, xcol)
    return grid, np.nanmedian(V, 0), np.nanpercentile(V, 25, axis=0), np.nanpercentile(V, 75, axis=0)


def _band(ax, x, med, lo, hi, algo, n):
    ax.plot(x, med, color=COLORS[algo], label=f"{NAMES[algo]} (n={n}, median)", lw=1.6)
    if n > 1:
        ax.fill_between(x, lo, hi, color=COLORS[algo], alpha=0.12)


def _style(ax, title, xlabel, ylabel, logx=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)


def _line(ax, x, m, s, algo, n):
    ax.plot(x, m, color=COLORS[algo], label=f"{NAMES[algo]} (n={n})", lw=1.6)
    if n > 1:
        ax.fill_between(x, m - s, m + s, color=COLORS[algo], alpha=0.12)


# --------------------------------------------------------------------------- figures
def fig_training_metric(key, ylabel, fname, window=20, hline=None, ylim=None):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=False)
    for ax, board in zip(axes, BOARDS):
        for algo in ALGO_ORDER:
            runs = load_runs(algo, board)
            if runs:
                x, m, s = mean_curve(runs, key, window)
                _line(ax, x, m, s, algo, len(runs))
        if hline is not None:
            ax.axhline(hline(board), color="k", ls="--", lw=1, label="BFS optimum" if key == "moves" else None)
        if ylim:
            ax.set_ylim(*ylim(board))
        _style(ax, f"{board} (N={BOARDS[board].n})", "training episode (log scale)", ylabel)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=130)
    plt.close(fig)


def fig_success():
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for j, board in enumerate(BOARDS):
        for algo in ALGO_ORDER:
            runs = load_runs(algo, board)
            if not runs:
                continue
            x, m, s = mean_curve(runs, "succ", 20)
            _line(axes[0, j], x, m * 100, s * 100, algo, len(runs))
            g, gm, gs = step_curve(runs, "eval_success")
            axes[1, j].plot(g, gm * 100, color=COLORS[algo], lw=1.6, label=f"{NAMES[algo]} (n={len(runs)})")
        axes[0, j].set_ylim(-5, 105)
        axes[1, j].set_ylim(-5, 105)
        _style(axes[0, j], f"{board}: training success rate (rolling 20)", "training episode (log)", "% episodes reaching the goal")
        _style(axes[1, j], f"{board}: greedy policy", "training episode (log)", "% of seeds whose greedy policy succeeds")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_success_vs_episodes.png", dpi=130)
    plt.close(fig)


def fig_q_vs_sarsa(per_run: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
    for j, board in enumerate(BOARDS):
        for algo in TABULAR:
            runs = load_runs(algo, board)
            if not runs:
                continue
            x, m, s = mean_curve(runs, "ret", 10)
            _line(axes[0, j], x, m, s, algo, len(runs))
            g, gmed, glo, ghi = step_median(runs, "eval_moves")
            _band(axes[1, j], g, gmed, glo, ghi, algo, len(runs))
        axes[1, j].axhline(bfs_min_rolls(BOARDS[board]), color="k", ls="--", lw=1, label="BFS optimum")
        axes[1, j].set_ylim(0, bfs_min_rolls(BOARDS[board]) * 3)
        _style(axes[0, j], f"{board}: training reward per episode", "training episode (log)", "total reward")
        _style(axes[1, j], f"{board}: greedy policy rolls (median, band = IQR)", "training episode (log)", "rolls to reach the goal")
        sub = per_run[(per_run.family == "tabular") & (per_run.board == board)]
        data = [sub[sub.algo == a]["conv_episode"].dropna().to_numpy() for a in TABULAR]
        bp = axes[2, j].boxplot(data, tick_labels=[NAMES[a] for a in TABULAR], showfliers=True, patch_artist=True)
        for patch, a in zip(bp["boxes"], TABULAR):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.4)
        axes[2, j].set_yscale("log")
        axes[2, j].set_title(f"{board}: episodes to convergence (n={len(sub) // 2} seeds each)")
        axes[2, j].set_ylabel("training episodes (log)")
        axes[2, j].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_q_vs_sarsa.png", dpi=130)
    plt.close(fig)


def fig_deep_timesteps(per_run: pd.DataFrame):
    deep = [a for a in ("dqn", "ppo", "a2c")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for j, board in enumerate(BOARDS):
        bfs = bfs_min_rolls(BOARDS[board])
        for algo in deep:
            runs = load_runs(algo, board)
            if not runs:
                continue
            xmin = min(r["ev"]["timesteps"].max() for r in runs)
            grid = np.arange(500, int(xmin) + 1, 500)
            V = np.vstack([np.interp(grid, r["ev"]["timesteps"], r["ev"]["eval_moves"]) for r in runs])
            _band(axes[0, j], grid, np.median(V, 0), np.percentile(V, 25, axis=0), np.percentile(V, 75, axis=0),
                  algo, len(runs))
        axes[0, j].axhline(bfs, color="k", ls="--", lw=1, label="BFS optimum")
        axes[0, j].set_ylim(0, bfs * 4)
        _style(axes[0, j], f"{board}: greedy policy rolls vs timesteps (median, band = IQR)", "environment timesteps", "rolls to reach the goal", logx=False)
        sub = per_run[(per_run.family == "deep") & (per_run.board == board)]
        for k, a in enumerate(deep):
            v = sub[sub.algo == a]["conv_timesteps"].dropna().to_numpy()
            axes[1, j].scatter(np.full(len(v), k) + np.linspace(-0.15, 0.15, max(len(v), 1))[: len(v)], v, color=COLORS[a], s=40)
            if len(v):
                axes[1, j].hlines(np.median(v), k - 0.3, k + 0.3, color="k")
        axes[1, j].set_xticks(range(len(deep)))
        axes[1, j].set_xticklabels([NAMES[a] for a in deep])
        axes[1, j].set_title(f"{board}: timesteps to convergence (bar = median)")
        axes[1, j].set_ylabel("environment timesteps")
        axes[1, j].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_deep_vs_timesteps.png", dpi=130)
    plt.close(fig)


def fig_convergence(per_run: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, board in zip(axes, BOARDS):
        sub = per_run[per_run.board == board]
        for k, a in enumerate(ALGO_ORDER):
            g = sub[sub.algo == a]
            v = g["conv_episode"].dropna().to_numpy()
            rng = np.random.default_rng(0)
            ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, color=COLORS[a], s=22, alpha=0.7)
            if len(v):
                ax.hlines(np.median(v), k - 0.3, k + 0.3, color="k")
            nc = int(g["conv_episode"].isna().sum())
            if nc:
                ax.text(k, ax.get_ylim()[0] if False else 0.6, f"{nc} not conv.", ha="center", fontsize=8, color="red")
        ax.set_yscale("log")
        ax.set_xticks(range(len(ALGO_ORDER)))
        ax.set_xticklabels([NAMES[a] for a in ALGO_ORDER])
        ax.set_title(f"{board}: training episodes to convergence")
        ax.set_ylabel("episodes (log); bar = median")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_convergence.png", dpi=130)
    plt.close(fig)


def fig_board_size(per_run: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = [BOARDS[b].n for b in BOARDS]
    for a in ALGO_ORDER:
        med, lo, hi = [], [], []
        for b in BOARDS:
            v = per_run[(per_run.board == b) & (per_run.algo == a)]["conv_episode"].dropna()
            med.append(v.median() if len(v) else np.nan)
            lo.append(v.min() if len(v) else np.nan)
            hi.append(v.max() if len(v) else np.nan)
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(sizes, med, yerr=[med - lo, hi - med], color=COLORS[a], marker="o", capsize=3, label=NAMES[a])
    ax.set_xticks(sizes)
    ax.set_yscale("log")
    ax.set_xlabel("board size N")
    ax.set_ylabel("training episodes to convergence (median, bars = min-max)")
    ax.set_title("Effect of board size on learning speed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "plot_board_size.png", dpi=130)
    plt.close(fig)


def main():
    per_run = build_per_run()
    if per_run.empty:
        print("No results found in results/.")
        return
    fig_training_metric("ret", "total reward per episode", "plot_reward_vs_episodes.png", window=20)
    fig_training_metric("moves", "rolls per episode", "plot_moves_vs_episodes.png", window=20,
                        hline=lambda b: bfs_min_rolls(BOARDS[b]), ylim=lambda b: (0, 6 * bfs_min_rolls(BOARDS[b])))
    fig_success()
    fig_q_vs_sarsa(per_run)
    if (per_run.family == "deep").any():
        fig_deep_timesteps(per_run)
    fig_convergence(per_run)
    fig_board_size(per_run)
    print("Saved PNG files to results/figures/:")
    for p in sorted(FIGURES.glob("plot_*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
