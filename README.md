# naive-rl
Naive implementations of CSE525 HWs

## Environment

### Install
Please check [Environment.ipynb](/Environment.ipynb)

### Test env
* Ubuntu 18.04
* Python 3.6.9
* torch 1.5.0
* numpy 1.18.3

## HW1 [Done]
### Algorithm
* Baseline: Vanilla Online Q Learning (without target network and replay buffer)
* Fitted Q Iteration
* DQN
* DQN-like-version of "SARSA"

### Test env
* InvertedPendulumMuJoCoEnv-v0
* HalfCheetahMuJoCoEnv-v0
* Breakout-v4-ram

## HW2 (working)
### Algorithm
* REINFORCE
* Advantage Actor Critic (A2C)

### Test env
* InvertedPendulumMuJoCoEnv-v0
* HalfCheetahMuJoCoEnv-v0

## HW3 (working)
### Algorithm
* Offline dynamics model learning
* Online dynamics model learning

### Test env
* Breakout-v4

