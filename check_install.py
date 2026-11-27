"""
check_install.py - quick test that your installation works.

It checks:
  1. Python and package versions.
  2. That torch is CPU-only (no GPU needed).
  3. That YOUR installed Stable-Baselines3 accepts a Discrete observation space
     with DQN, PPO and A2C (SB3 one-hot encodes Discrete observations internally).
Uses a tiny throw-away environment, so it does NOT depend on env.py.
"""
import sys
import time


def main():
    print("Python:", sys.version.split()[0])
    import gymnasium as gym
    import matplotlib
    import numpy as np
    import pandas as pd
    import stable_baselines3 as sb3
    import torch
    from gymnasium import spaces

    print("gymnasium:", gym.__version__)
    print("stable-baselines3:", sb3.__version__)
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "(not needed)")
    print("numpy:", np.__version__, "| pandas:", pd.__version__, "| matplotlib:", matplotlib.__version__)

    class TinyEnv(gym.Env):
        """5 squares in a row, Discrete(5) observation, Discrete(6) action, like our board game."""

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Discrete(5)
            self.action_space = spaces.Discrete(6)
            self.pos = 0
            self.t = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.pos, self.t = 0, 0
            return self.pos, {}

        def step(self, action):
            self.t += 1
            self.pos = min(4, self.pos + (int(action) % 3))
            terminated = self.pos == 4
            truncated = (not terminated) and self.t >= 20
            return self.pos, (10.0 if terminated else -1.0), terminated, truncated, {}

    ok = True
    for name in ["DQN", "PPO", "A2C"]:
        try:
            Algo = getattr(sb3, name)
            env = TinyEnv()
            model = Algo("MlpPolicy", env, seed=0, device="cpu", verbose=0)
            t0 = time.time()
            model.learn(total_timesteps=600)
            obs, _ = env.reset(seed=0)
            action, _ = model.predict(obs, deterministic=True)
            assert env.action_space.contains(int(action))
            print(f"[ok]   {name}: trained 600 steps in {time.time() - t0:.1f}s, predict() works")
        except Exception as e:  # report honestly, do not hide failures
            ok = False
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    print("\nRESULT:", "everything works" if ok else "SOMETHING FAILED - paste this output to me")


if __name__ == "__main__":
    main()
