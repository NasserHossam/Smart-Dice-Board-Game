"""
paths.py - every folder the project reads from or writes to, defined in one place.

    results/tabular/        Q-Learning and SARSA runs (tabular.py)
    results/deep_rl/        DQN, PPO and A2C runs (sb3_train.py)
    results/tables/         final tables, convergence tables, baselines, pilot timing
    results/experiments/    reward-function and layout experiments (experiments.py)
    results/figures/        all main-project figures (plots.py, experiments.py)
    results_challenges/chK/ outputs of challenges/challengeK_*.py

Scripts in challenges/ and demos/ put src/ on the import path with one line at the top, so every script
can be run from the project folder, e.g.  python challenges/challenge1_random_dice.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABULAR = RESULTS / "tabular"
DEEP = RESULTS / "deep_rl"
TABLES = RESULTS / "tables"
EXPERIMENTS = RESULTS / "experiments"
FIGURES = RESULTS / "figures"
CHALLENGES = ROOT / "results_challenges"


def challenge_dir(k: int) -> Path:
    """Output folder of Challenge k (created on first use)."""
    d = CHALLENGES / f"ch{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d
