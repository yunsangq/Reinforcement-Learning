import gym
from gym import wrappers
import numpy as np

env = gym.make("FrozenLake-v0")
env = wrappers.Monitor(env, "./results", force=True)

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 5000
rList = []
gamma = 0.99
alpha = 0.85

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
    while not done:
        new_state, reward, done, _ = env.step(action)
        new_action = np.argmax(Q[new_state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
        rAll += reward
        state = new_state
        action = new_action
    rList.append(rAll)
    if i % 500 == 0 and i is not 0:
        print("Success rate: " + str(sum(rList) / i))

print("Success rate: " + str(sum(rList)/num_episodes))

env.close()