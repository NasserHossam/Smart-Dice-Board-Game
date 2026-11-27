"""
pilot_timing.py - measure how fast DQN, PPO and A2C run on YOUR machine, before choosing timesteps.

It trains each algorithm for a small number of steps (default 3000), including the cost of the
periodic greedy evaluation, and prints steps/second plus a NAIVE linear estimate of the time for
longer runs. Nothing is saved except results/pilot_timing.csv. Pilot learning results are not
for the report; only the speed is used.

Caveat: the estimate assumes constant speed. DQN only starts gradient updates after 500 steps and
PPO/A2C cost per step can change, so treat the numbers as rough and always keep --time-limit.

Usage:  python pilot_timing.py
        python pilot_timing.py --timesteps 5000 --boards N30 N100
"""
import argparse

import pandas as pd

from env import BOARDS
from sb3_train import ALGOS, RESULTS_DIR, train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3000)
    p.add_argument("--boards", nargs="+", default=["N30", "N100"], choices=list(BOARDS))
    p.add_argument("--time-limit", type=float, default=300, help="safety cap per pilot run (seconds)")
    a = p.parse_args()

    rows = []
    for b in a.boards:
        for algo in ALGOS:
            s = train_one(b, algo, a.timesteps, seed=0, time_limit=a.time_limit, out_dir=None, verbose=False)
            rows.append({"board": b, "algo": algo, "timesteps_run": s["timesteps_run"],
                         "elapsed_s": s["elapsed_s"], "steps_per_sec": s["steps_per_sec"],
                         "episodes": s["episodes_run"], "stopped_by_time": s["stopped_by_time_limit"]})
            print(f"{b:<5}{algo:<5}{s['timesteps_run']:>7} steps in {s['elapsed_s']:>6.1f}s -> "
                  f"{s['steps_per_sec']:>7.1f} steps/s", flush=True)

    df = pd.DataFrame(rows)
    for target in (20_000, 50_000, 100_000, 200_000):
        df[f"est_min_{target // 1000}k"] = (target / df.steps_per_sec / 60).round(1)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "pilot_timing.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nNAIVE estimate of minutes PER RUN (linear extrapolation of the pilot speed):")
    print(df.drop(columns=["episodes", "stopped_by_time"]).to_string(index=False))
    print("\nSaved results/pilot_timing.csv")


if __name__ == "__main__":
    main()
