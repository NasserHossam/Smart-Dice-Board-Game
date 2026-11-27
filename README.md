# Smart Dice Board Game: Reinforcement Learning

A Snakes-and-Ladders style board game solved with reinforcement learning.
The agent chooses the dice value (1 to 6) each turn, starts on square 1 and must reach square N.

* **Group A (Stable-Baselines3):** DQN, PPO, A2C
* **Group B (written in NumPy):** tabular Q-Learning and SARSA, both epsilon-greedy
* Boards: N30 (3 ladders, 3 snakes), N50 (5, 5), N100 (8, 8), fixed layouts documented in `env.py`
* CPU only, single process, Python 3.10 or newer (tested with 3.12.2, SB3 2.9.2, Gymnasium 1.4.0)

## Environment rules (`env.py`)

* State: current square (`Discrete(N)`, observation = square - 1). Action: `Discrete(6)`, action a = roll a + 1.
* A ladder moves the agent forward and a snake moves it back, immediately, never chained.
* **Invalid move (roll past square N): the agent stays, gets -10, and the roll still counts.**
* Rewards: goal +100, ladder +20, normal square -1, snake -20, invalid -10.
* An episode is cut after 5 * N rolls and counts as a failure.
* `bfs_min_rolls()` gives the true minimum number of rolls: 4 (N30), 5 (N50), 10 (N100).

## Setup (Windows, Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python check_install.py
python env.py
```

`check_install.py` trains DQN, PPO and A2C for 600 steps each to confirm they accept the Discrete observation space.
`env.py` runs the self-test and prints the random-policy baseline.

## Files

| File | Purpose |
|---|---|
| `env.py` | Gymnasium environment, three board layouts, BFS optimum, self-test |
| `tabular.py` | Q-Learning and SARSA with epsilon decay and per-episode logging |
| `sb3_train.py` | DQN, PPO, A2C training with greedy-policy checkpoints |
| `pilot_timing.py` | Short timing run used to choose budgets |
| `analyze_tabular.py` | Convergence metrics for all saved tabular runs |
| `evaluate.py` | Final tables and random baseline |
| `unified_convergence.py` | One convergence rule for all five algorithms, in episodes and environment steps |
| `experiments.py` | Reward-function and snake/ladder-layout experiments (tabular) |
| `plots.py` | All required figures |
| `demo.py` | Play one game with a random, saved, or BFS policy |
| `demo_learning.py` | Watch Q-Learning and SARSA learn over episodes |
| `results/` | All CSV, JSON, NPY and PNG outputs |

## How the results were produced

The settings of every run are stored in `results/*_summary.json`. Check that these commands match what you ran:

```
# tabular: 2000 episodes; seeds 0-4, then seeds 5-19 with the tag _extra
python tabular.py --board all --algo both --episodes 2000 --seeds 0 1 2 3 4
python tabular.py --board all --algo both --episodes 2000 --seeds 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --tag _extra

# deep RL: 3 seeds, budget per board
python sb3_train.py --board N30  --algo all --timesteps 30000  --seeds 0 1 2
python sb3_train.py --board N50  --algo all --timesteps 50000  --seeds 0 1 2
python sb3_train.py --board N100 --algo all --timesteps 100000 --seeds 0 1 2

# tables and plots
python analyze_tabular.py
python evaluate.py
python unified_convergence.py
python plots.py

# experiments
python experiments.py --which reward --board N50 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 2000
python experiments.py --which layout --seeds 0 1 2 3 4 --episodes 1000 --time-limit 120
```

Both training scripts accept `--time-limit SECONDS` per run; a run that hits it stops and saves what it has.
No run in the reported results hit the limit.

## Demos

```
python demo.py --board N30 --source random --seed 3                              # before training
python demo_learning.py                                                          # learning over episodes, Q-Learning vs SARSA
python demo.py --board N100 --source tabular --algo sarsa --seed 0               # after training
python demo.py --board N30 --source file --file results\sb3_ppo_N30_seed0_policy.npy   # a trained PPO policy
```

`demo.py --source bfs` (also the default) plays the BFS reference path. It is a ruler, not a learned policy.

## Convergence criterion

A checkpoint is good if greedy success is at least 95% and mean greedy rolls are at most 1.10 times the BFS minimum.
A run converged at episode E if E and every later checkpoint until the end of training are good.
Otherwise it is reported as not converged. The tables use this rule for all five algorithms.

## Honest notes

* The environment and the greedy policy are deterministic, so one greedy game is the whole evaluation. Success is 0% or 100% per run.
* A random policy already reaches the goal every time on these boards, so rolls and return are the informative metrics.
* Tabular results use 20 seeds, deep RL uses 3. Hyperparameters were not tuned.
* The report lists all limitations.
