"""
sb3_train.py - GROUP A: three deep RL algorithms from Stable-Baselines3 on the Smart Dice env.

  DQN : value-based. A neural net estimates Q(s, a). Learns from a replay buffer of past steps
        (breaks correlation between consecutive samples) and a slowly-updated target network
        (stabilises the bootstrap target). Exploration = epsilon-greedy, decayed linearly.
  PPO : policy-gradient, actor-critic. Updates the policy a little at a time: the probability
        ratio new/old is CLIPPED so one update cannot move the policy too far. Stable, popular default.
  A2C : synchronous advantage actor-critic. Actor picks actions, critic estimates state value;
        the update uses the advantage (how much better than expected) on short rollouts.
        Simpler and faster per update than PPO, but less stable.

All three accept Discrete observations (SB3 one-hot encodes them) and Discrete(6) actions.
Single process only (no subprocess vector envs), so this runs the same locally and in a notebook.

HONEST NOTES FOR THE REPORT
  * Hyper-parameters below are reasonable defaults for a tiny problem, NOT tuned.
  * Rewards are multiplied by --reward-scale (default 0.1) ONLY for the neural networks, because
    raw rewards up to +100 make value targets large. Positive scaling does not change which policy
    is optimal. Every return that is logged or reported uses the ORIGINAL rewards.
  * Evaluation: the env and a deterministic policy are both deterministic, so every greedy game
    is identical. We therefore build the greedy policy table (best action for each of the N squares)
    in one batch, play it once, and that single game stands for all 100 identical games.
    Success is therefore 0% or 100% at each checkpoint. Say this plainly in the report.
  * Convergence (same rule as tabular): from checkpoint E until the END of training every checkpoint
    has success >= 95% AND mean rolls <= 1.10 * BFS. Not met -> "not converged".

Example:  python sb3_train.py --board N30 --algo all --timesteps 20000 --seeds 0 --time-limit 600
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence unrelated TensorFlow/protobuf noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from env import BOARDS, SmartDiceEnv, bfs_min_rolls

torch.set_num_threads(1)  # tiny networks: one thread is faster than many
RESULTS_DIR = Path(__file__).parent / "results"
ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


# --------------------------------------------------------------------------- hyper-parameters
def make_model(algo: str, env, seed: int):
    net = dict(net_arch=[64, 64])
    if algo == "dqn":
        hp = dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=500, batch_size=64, gamma=0.99,
                  train_freq=4, gradient_steps=1, target_update_interval=250,
                  exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.4)
    elif algo == "ppo":
        hp = dict(learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10, gamma=0.99,
                  gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    elif algo == "a2c":
        hp = dict(learning_rate=7e-4, n_steps=16, gamma=0.99, gae_lambda=1.0, ent_coef=0.01)
    else:
        raise ValueError(f"unknown algo {algo}")
    model = ALGOS[algo]("MlpPolicy", env, policy_kwargs=net, seed=seed, device="cpu", verbose=0, **hp)
    return model, {**hp, "net_arch": net["net_arch"]}


class ScaledReward(gym.Wrapper):
    """Scales the reward the network sees and keeps the original in info['raw_reward']."""

    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = r
        return obs, r * self.scale, terminated, truncated, info


# --------------------------------------------------------------------------- greedy evaluation
def policy_table(model, n: int) -> np.ndarray:
    """Greedy action (0..5) for every observation 0..n-1. Batched; checked against single calls."""
    obs = np.arange(n)
    actions, _ = model.predict(obs, deterministic=True)
    table = np.asarray(actions).reshape(-1)
    if table.shape[0] != n or any(int(model.predict(int(s), deterministic=True)[0]) != int(table[s])
                                  for s in (0, n // 2, n - 1)):
        table = np.array([int(model.predict(int(s), deterministic=True)[0]) for s in range(n)])
    return table


def play_table(env: SmartDiceEnv, table: np.ndarray):
    """One greedy game. Returns (success, rolls, original_return)."""
    obs, _ = env.reset()
    done, ret, term = False, 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(int(table[obs]))
        ret += r
        done = term or trunc
    return bool(term), env.steps, ret


def stable_start_index(ok: np.ndarray):
    """Index of the first checkpoint from which every later checkpoint is ok (None if the last is not)."""
    if len(ok) == 0 or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i


# --------------------------------------------------------------------------- callback
class LogCallback(BaseCallback):
    def __init__(self, board: str, eval_every: int, time_limit, total_timesteps: int,
                 rewards=None, max_steps=None, verbose_name=""):
        super().__init__(verbose=0)
        self.eval_env = SmartDiceEnv(board, rewards=rewards, max_steps=max_steps)
        self.n = self.eval_env.n
        self.eval_every, self.time_limit, self.total = eval_every, time_limit, total_timesteps
        self.name = verbose_name
        self.episodes, self.evals = [], []
        self.ep_ret, self.ep_len = 0.0, 0
        self.stopped_by_time = False
        self.last_table = None
        self._next_print = max(1, total_timesteps // 10)

    def _on_training_start(self):
        self.t0 = time.time()

    def _do_eval(self):
        table = policy_table(self.model, self.n)
        ok, moves, ret = play_table(self.eval_env, table)
        recent = self.episodes[-20:]
        self.last_table = table
        row = {
            "timesteps": self.num_timesteps, "episodes": len(self.episodes),
            "wall_s": round(time.time() - self.t0, 2),
            "eval_success": 1.0 if ok else 0.0, "eval_moves": float(moves), "eval_return": float(ret),
            "train_return_last20": float(np.mean([e["return"] for e in recent])) if recent else np.nan,
            "train_moves_last20": float(np.mean([e["moves"] for e in recent])) if recent else np.nan,
            "train_success_last20": float(np.mean([e["success"] for e in recent])) if recent else np.nan,
            "exploration_rate": float(getattr(self.model, "exploration_rate", np.nan)),
        }
        self.evals.append(row)
        return row

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.ep_ret += float(info["raw_reward"])
        self.ep_len += 1
        if bool(self.locals["dones"][0]):
            self.episodes.append({"episode": len(self.episodes) + 1, "timesteps": self.num_timesteps,
                                  "return": self.ep_ret, "moves": self.ep_len,
                                  "success": int(info["square"] == self.n)})
            self.ep_ret, self.ep_len = 0.0, 0
        if self.num_timesteps % self.eval_every == 0:
            row = self._do_eval()
            if self.num_timesteps >= self._next_print:
                self._next_print += max(1, self.total // 10)
                print(f"  [{self.name}] step {row['timesteps']:>7}/{self.total}  episodes={row['episodes']:>5}  "
                      f"greedy: success={row['eval_success'] * 100:3.0f}%  rolls={row['eval_moves']:6.1f}  "
                      f"return={row['eval_return']:7.1f}  t={row['wall_s']:6.1f}s", flush=True)
        if self.time_limit is not None and time.time() - self.t0 > self.time_limit:
            self.stopped_by_time = True
            return False
        return True

    def _on_training_end(self):
        if not self.evals or self.evals[-1]["timesteps"] != self.num_timesteps:
            self._do_eval()


# --------------------------------------------------------------------------- one run
def train_one(board: str, algo: str, timesteps: int, seed: int, eval_every: int = 500,
              time_limit: float | None = None, reward_scale: float = 0.1,
              rewards: dict | None = None, max_steps: int | None = None,
              out_dir: Path | None = RESULTS_DIR, tag: str = "", verbose: bool = True):
    """Train one deep RL agent. Saves logs if out_dir is not None. Returns the summary dict."""
    env = ScaledReward(SmartDiceEnv(board, rewards=rewards, max_steps=max_steps), reward_scale)
    model, hp = make_model(algo, env, seed)
    cb = LogCallback(board, eval_every, time_limit, timesteps, rewards, max_steps, f"{algo} {board} seed{seed}")
    bfs = bfs_min_rolls(BOARDS[board])
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    ev = pd.DataFrame(cb.evals)
    ok_loose = ((ev.eval_success >= 0.95) & (ev.eval_moves <= 1.10 * bfs)).to_numpy()
    ok_strict = ((ev.eval_success >= 0.95) & (ev.eval_moves <= bfs + 1e-9)).to_numpy()
    i_c, i_o = stable_start_index(ok_loose), stable_start_index(ok_strict)
    last = ev.iloc[-1]
    summary = {
        "algo": algo, "board": board, "seed": seed, "bfs_optimum": bfs,
        "timesteps_planned": timesteps, "timesteps_run": int(model.num_timesteps),
        "episodes_run": len(cb.episodes), "stopped_by_time_limit": cb.stopped_by_time,
        "elapsed_s": round(elapsed, 2), "steps_per_sec": round(model.num_timesteps / max(elapsed, 1e-9), 1),
        "converged": i_c is not None,
        "convergence_timesteps": None if i_c is None else int(ev.timesteps.iloc[i_c]),
        "convergence_episode": None if i_c is None else int(ev.episodes.iloc[i_c]),
        "optimal_timesteps": None if i_o is None else int(ev.timesteps.iloc[i_o]),
        "optimal_episode": None if i_o is None else int(ev.episodes.iloc[i_o]),
        "final_eval_success": float(last.eval_success), "final_eval_moves": float(last.eval_moves),
        "final_eval_return": float(last.eval_return),
        "reward_scale": reward_scale, "eval_every": eval_every, "hyperparameters": hp,
        "rewards": rewards or "default",
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sb3_{algo}_{board}_seed{seed}{tag}"
        pd.DataFrame(cb.episodes).to_csv(out_dir / f"{stem}_episodes.csv", index=False)
        ev.to_csv(out_dir / f"{stem}_evals.csv", index=False)
        np.save(out_dir / f"{stem}_policy.npy", cb.last_table)   # action index per obs (roll = a+1, square = obs+1)
        model.save(str(out_dir / f"{stem}_model"))
        with open(out_dir / f"{stem}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        summary["stem"] = stem
    return summary


# --------------------------------------------------------------------------- command line
def main():
    p = argparse.ArgumentParser(description="Train SB3 DQN / PPO / A2C on the Smart Dice board game")
    p.add_argument("--board", default="N30", help="N30, N50, N100 or all")
    p.add_argument("--algo", default="all", help="dqn, ppo, a2c or all")
    p.add_argument("--timesteps", type=int, required=True, help="environment steps per run (run the pilot first)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--time-limit", type=float, default=None, help="max seconds PER RUN; saves what it has")
    p.add_argument("--eval-every", type=int, default=500, help="timesteps between greedy evaluations")
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--tag", default="")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip runs whose summary file already exists (use after an interrupted session)")
    a = p.parse_args()

    boards = list(BOARDS) if a.board == "all" else [a.board]
    algos = list(ALGOS) if a.algo == "all" else [a.algo]
    for b in boards:
        for algo in algos:
            for seed in a.seeds:
                stem = f"sb3_{algo}_{b}_seed{seed}{a.tag}"
                if a.skip_existing and (RESULTS_DIR / f"{stem}_summary.json").exists():
                    print(f"==> {stem}: already finished, skipping", flush=True)
                    continue
                s = train_one(b, algo, a.timesteps, seed, eval_every=a.eval_every, time_limit=a.time_limit,
                              reward_scale=a.reward_scale, tag=a.tag)
                status = (f"converged at episode {s['convergence_episode']} (step {s['convergence_timesteps']})"
                          if s["converged"] else "NOT converged")
                if s["stopped_by_time_limit"]:
                    status += " (stopped by time limit)"
                print(f"==> {s['stem']}: {status}; final greedy rolls={s['final_eval_moves']:.0f} "
                      f"(BFS {s['bfs_optimum']}); {s['timesteps_run']}/{s['timesteps_planned']} steps, "
                      f"{s['episodes_run']} episodes in {s['elapsed_s']}s ({s['steps_per_sec']} steps/s)\n", flush=True)


if __name__ == "__main__":
    main()
