# Smart Dice Board Game: Reinforcement Learning

A Snakes-and-Ladders style board game solved with reinforcement learning. The agent chooses the dice value (1 to 6) each turn, starts on square 1 and must reach square N.

- **Group A (Stable-Baselines3):** DQN, PPO, A2C
- **Group B (written in NumPy):** tabular Q-Learning and SARSA, both epsilon-greedy
- Boards: N30 (3 ladders, 3 snakes), N50 (5, 5), N100 (8, 8), fixed layouts documented in `src/env.py`
- **Advanced challenges:** all six: random dice, a one-time snake shield, a dynamic board, a two-player race, DQN vs tabular Q-Learning on larger boards, and reward shaping (see the section below)
- CPU only, single process, Python 3.10 or newer (tested with 3.12.2, SB3 2.9.2, Gymnasium 1.4.0)

## Environment rules (`src/env.py`)

- State: current square (`Discrete(N)`, observation = square - 1). Action: `Discrete(6)`, action a = roll a + 1.
- A ladder moves the agent forward and a snake moves it back, immediately, never chained.
- **Invalid move (roll past square N): the agent stays, gets -10, and the roll still counts.**
- Rewards: goal +100, ladder +20, normal square -1, snake -20, invalid -10.
- An episode is cut after 5 * N rolls and counts as a failure.
- `bfs_min_rolls()` gives the true minimum number of rolls: 4 (N30), 5 (N50), 10 (N100).

## Setup (Windows, Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/check_install.py
python src/env.py
```

`src/check_install.py` trains DQN, PPO and A2C for 600 steps each to confirm they accept the Discrete observation space. `src/env.py` runs the self-test and prints the random-policy baseline.

## Project structure

```
Smart Dice Board Game/
├── README.md, requirements.txt
├── src/                   core code (run every script from the project folder)
│   ├── env.py             Gymnasium environment, three board layouts, BFS optimum, self-test
│   ├── paths.py           every results folder, defined in one place
│   ├── tabular.py         Q-Learning and SARSA with epsilon decay and per-episode logging
│   ├── sb3_train.py       DQN, PPO, A2C training with greedy-policy checkpoints
│   ├── check_install.py   install check (trains DQN, PPO, A2C for 600 steps)
│   ├── pilot_timing.py    short timing run used to choose budgets
│   ├── analyze_tabular.py convergence metrics for all saved tabular runs
│   ├── evaluate.py        final tables and random baseline
│   ├── unified_convergence.py  one convergence rule for all five algorithms
│   ├── experiments.py     reward-function and snake/ladder-layout experiments (tabular)
│   └── plots.py           all required figures
├── challenges/            challenge1_random_dice.py ... challenge6_reward_shaping.py
├── demos/                 demo.py (play one game), demo_learning.py (watch Q-Learning and SARSA learn)
├── app/                   live_game.py: watch any player live in the browser (Streamlit)
├── results/
│   ├── tabular/           Q-Learning and SARSA runs: *_log.csv, *_Q.npy, *_summary.json
│   ├── deep_rl/           DQN, PPO, A2C runs: *_episodes.csv, *_evals.csv, *_policy.npy, *_summary.json
│   ├── tables/            final tables, convergence tables, random baseline, pilot timing
│   ├── experiments/       reward-function and layout experiment tables
│   └── figures/           all main-project figures (PNG)
├── results_challenges/
│   └── ch1/ ... ch6/      outputs of each challenge
└── docs/                  the report (.docx)
```

| Challenge script                         | What it does                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| `challenges/challenge1_random_dice.py`   | Challenge 1: random dice, exact analysis and TD value learning              |
| `challenges/challenge2_shield.py`        | Challenge 2: one-time snake shield on random dice                           |
| `challenges/challenge3_dynamic_board.py` | Challenge 3: layouts change between episodes, generalization and adaptation |
| `challenges/challenge4_two_players.py`   | Challenge 4: agent vs a rule-based opponent                                 |
| `challenges/challenge5_dqn_vs_tabular.py`| Challenge 5: DQN vs tabular Q-Learning on larger boards (N200, N400)        |
| `challenges/challenge6_reward_shaping.py`| Challenge 6: reward shaping (sparse, potential-based, progress bonus)       |

Trained SB3 model files (`*_model.zip`) are not in the repository; the greedy policy of every run is saved as `*_policy.npy`.

## How the results were produced

The settings of every run are stored in `results/tabular/*_summary.json` and `results/deep_rl/*_summary.json`. Check that these commands match what you ran:

```
# tabular: 2000 episodes; seeds 0-4, then seeds 5-19 with the tag _extra
python src/tabular.py --board all --algo both --episodes 2000 --seeds 0 1 2 3 4
python src/tabular.py --board all --algo both --episodes 2000 --seeds 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --tag _extra

# deep RL: 3 seeds, budget per board
python src/sb3_train.py --board N30  --algo all --timesteps 30000  --seeds 0 1 2
python src/sb3_train.py --board N50  --algo all --timesteps 50000  --seeds 0 1 2
python src/sb3_train.py --board N100 --algo all --timesteps 100000 --seeds 0 1 2

# tables and plots
python src/analyze_tabular.py
python src/evaluate.py
python src/unified_convergence.py
python src/plots.py

# experiments
python src/experiments.py --which reward --board N50 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 2000
python src/experiments.py --which layout --seeds 0 1 2 3 4 --episodes 1000 --time-limit 120
```

Both training scripts accept `--time-limit SECONDS` per run; a run that hits it stops and saves what it has. No run in the reported results hit the limit.

## Demos

```
python demos/demo.py --board N30 --source random --seed 3                              # before training
python demos/demo_learning.py                                                          # learning over episodes, Q-Learning vs SARSA
python demos/demo.py --board N100 --source tabular --algo sarsa --seed 0               # after training
python demos/demo.py --board N30 --source file --file results\deep_rl\sb3_ppo_N30_seed0_policy.npy   # a trained PPO policy
```

`demos/demo.py --source bfs` (also the default) plays the BFS reference path. It is a ruler, not a learned policy.

## Live game in the browser

```
streamlit run app/live_game.py
```

This opens a page at http://localhost:8501 where you pick a board (N30, N50, N100, or the Challenge 5 boards N200 and N400) and a player (BFS shortest path, random rolls, or any trained Q-Learning, SARSA, DQN, PPO or A2C seed), then press Play or Step to watch the game roll by roll on a drawn board. It shows each roll, the reward, the total return, and the rolls compared with the BFS minimum. The page uses the real environment from `src/env.py` and only reads the saved policies; it never trains or changes anything.

## Convergence criterion

A checkpoint is good if greedy success is at least 95% and mean greedy rolls are at most 1.10 times the BFS minimum. A run converged at episode E if E and every later checkpoint until the end of training are good. Otherwise it is reported as not converged. The tables use this rule for all five algorithms.

## Advanced challenges

Each challenge is a separate script with its own self-test, and all outputs go to `results_challenges/chK/` (K = 1 to 6). The main project files and their results are not touched. Every script prints its self-test first and stops if it fails. Q-Learning is used in Challenges 3 and 4. Challenges 1, 2 and 6 use Q-Learning and SARSA. Challenge 5 compares Q-Learning with DQN.

```
python challenges/challenge1_random_dice.py --episodes 20000 --seeds 0 1 2 3 4 --mc-games 20000 --time-limit 120
python challenges/challenge2_shield.py --episodes 200000 --seeds 0 1 2 3 4 --mc-games 10000 --time-limit 300
python challenges/challenge3_dynamic_board.py --episodes 30000 --seeds 0 1 2 3 4
python challenges/challenge4_two_players.py --episodes 40000 --seeds 0 1 2
python challenges/challenge4_two_players.py --episodes 40000 --seeds 0 1 2 --first opponent
python challenges/challenge5_dqn_vs_tabular.py --workers 6
python challenges/challenge6_reward_shaping.py --workers 2
```

### Challenge 1: random dice

The environment rolls the dice, so the agent has no decisions. The game is a Markov reward process, and Q-Learning and SARSA reduce to the same value-learning rule. Win probability and expected game length are properties of the board. The script solves the Markov chain exactly, simulates it, and checks that TD learning recovers the exact values.

- Exact expected game length: 14.83 rolls (N30), 22.06 (N50), 37.59 (N100).
- Simulation agrees with the exact values within sampling error. A few of the gaps are about 2 standard errors, which is plausible with this many comparisons.
- The learned values stay a few reward points away from the exact ones with a constant learning rate. The cause was not tested in the committed runs.

### Challenge 2: one-time snake shield

The shield only has value when the dice are random, so this challenge uses random dice. The state is (square, shield left, rolled value). The action is "use the shield or not". The shield cancels a snake: the agent stays on the snake head, gets the normal-square reward instead of -20, and the shield is spent. The optimal policy is found exactly by policy iteration (discounted return, gamma 0.99), and Q-Learning and SARSA are scored exactly against it.

| Board | Expected return without shield | With the optimal shield policy | Expected rolls |
| ----- | ------------------------------ | ------------------------------ | -------------- |
| N30   | 52.86                          | 65.57                          | 14.83 to 12.36 |
| N50   | 46.46                          | 62.92                          | 22.06 to 18.35 |
| N100  | 35.42                          | 57.80                          | 37.59 to 32.58 |

- On N30 and N50, "use the shield on the first snake you would hit" is already optimal. On N100 the optimal policy skips the shield at the snake 34 to 19 and uses it in 42 of the 48 decision states (see `ch2_optimal_policy_N100.csv`).
- Learned policies: final regret (optimal minus learned discounted value) is 0.000 for the median run on N30 for both algorithms. On N100 it is 0.253 (Q-Learning) and 0.391 (SARSA). Over the last 20% of training, only about 8% to 10% of N100 checkpoints were within the 0.1 tolerance, so most N100 runs are reported as not converged.
- The "converged at episode" column mostly reflects the epsilon schedule and should not be read as learning speed.

### Challenge 3: dynamic board

Each episode starts on a layout drawn from a pool (N=50, 5 ladders, 5 snakes, jump length 8 to 16). The agent still chooses the dice. Two state designs are compared: position-only (the square) and a local view (what each of the next 6 rolls would do). Agents are trained on a pool of M layouts and tested on 200 held-out layouts. "Excess rolls" means rolls used minus the BFS minimum, counted over games that reached the goal.

- Baselines on the held-out layouts: a hand-written rule that uses only the local view has 0.175 excess rolls (100% success). Uniformly random rolls have 14.99.
- Position-only agents reach the goal on about half of the held-out layouts for M of 5 or more (median 46.5% to 52.5%) and 84.5% for M=1.
- The local-view agent reaches the goal on 99% or more, with median excess 2.17 rolls at M=200. It is clearly worse than the hand-written rule.
- Adaptation (100 cases): a position-only agent moved to a new board plays with median excess 7.60 and 89% success. After retraining it recovered in 86% of cases, in a median of 90 episodes (median over the recovered cases). The local-view agent trained on 200 layouts, with no retraining, had 100% success and median excess 2.00.

### Challenge 4: agent vs a rule-based opponent

Players alternate turns and the first to reach square N wins. The agent chooses its dice. Opponents: random rolls, a hand-written rule with 30% random moves, and a shortest-path (BFS) opponent. A second variant, **bump**, is my own addition and is not part of the original rules: landing on the other player's square sends that player back to square 1. Win probabilities are computed exactly, and the optimal agent is found by value and policy iteration.

- Without bump, the players do not interact. The BFS agent is exactly optimal for every opponent (the self-test checks this), so the opponent's position is not useful information.
- Moving first matters. When the agent moves first it wins about 100% of games against every opponent. When the opponent moves first, the optimal win probability on N30 is 0.989 (random opponent), 0.508 (rule) and 0.000 (BFS).
- With bump, the optimal agent wins 1.000 in every case. The BFS agent still loses every game to the BFS opponent.
- Q-Learning (3 seeds, opponent first, with bump) reached 1.000 against the BFS opponent on both boards. Against the rule opponent on N50 the median was 0.960 against an optimal value of 1.000.
- In two no-bump runs (N30 and N50 against the rule opponent) one seed finished clearly below optimal (0.427 and 0.622). The cause was not tested.

### Challenge 5: DQN vs tabular Q-Learning on larger boards

Two generated boards with the same jump density as N100: N200 (16 ladders, 16 snakes) and N400 (32, 32), layouts in `results_challenges/ch5/ch5_boards.json`. Q-Learning uses the main settings (2,000 episodes, 10 seeds); DQN uses the main untuned hyperparameters with 1,000 x N steps (3 seeds).

- On N200 the policy with the best discounted return uses 18 rolls (BFS 17): a one-roll detour collects an extra ladder. DQN found it in all 3 seeds after a median of 197,500 steps. Q-Learning with 2,000 episodes ended on 19 to 21 rolls and none of the 10 runs converged.
- With 10,000 episodes (`--budget-check 10000`) Q-Learning found the 18-roll policy in 10 of 10 seeds, after a median of 345 episodes (about 26,000 steps). The 2,000-episode schedule stopped exploring too early.
- On N400 both converged in every seed (Q-Learning median 24,231 steps, DQN 122,500).
- DQN needed 5 to 31 times more environment steps than Q-Learning, and its runs took about 70 to 90 times longer.

### Challenge 6: reward shaping

Part 1 is the reward-function experiment in `src/experiments.py`. Part 2 compares five learning rewards (default, sparse, sparse + potential-based shaping, default + potential-based shaping, and a forward-progress bonus) with Q-Learning and SARSA on N50, N100 and N200 (10 seeds, 2,000 episodes). Every result is measured under the original rewards.

- Potential-based shaping (Phi = progress along the board) made learning faster without changing the final policy, for example SARSA on N100 from a median of 40.5 to 10.5 episodes.
- The sparse reward was 2 to 3 times slower and, because nothing pays for ladders, picked equally short paths with a lower original-reward return.
- The progress bonus is not potential-based. Value iteration shows its optimal policy never finishes, and Q-Learning exploited the ladder-snake loop in all 30 runs (0 games finished). The loop appeared only after a few hundred to more than a thousand episodes.

## Honest notes

- The environment and the greedy policy are deterministic, so one greedy game is the whole evaluation. Success is 0% or 100% per run.
- A random policy already reaches the goal every time on these boards, so rolls and return are the informative metrics.
- Tabular results use 20 seeds, deep RL uses 3. Hyperparameters were not tuned.
- Challenge results use 3 to 10 seeds, so ranges overlap and no ranking between algorithms is claimed.
- The regret tolerance in Challenge 2 (0.1), the layout family in Challenge 3, the local-view design, the hand-written rule, and the bump rule in Challenge 4 are choices made for this project.
- The report lists all limitations.
