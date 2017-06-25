import gym
from gym import wrappers
import random
from utils.TileCoding import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000

numtilings = 8
maxtiles = 2048
thetas = np.zeros(maxtiles)
alpha = 0.01
gamma = 0.98
epsilon = 0.2

hashTable = IHT(maxtiles)


# get indices of active tiles for given state and action
def getActiveTiles(position, velocity, action):
    global hashTable
    global env
    # I think positionScale * (position - position_min) would be a good normalization.
    # However positionScale * position_min is a constant, so it's ok to ignore it.
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    activeTiles = tiles(hashTable, numtilings,
                        [numtilings * position / (max_position - min_position), numtilings * velocity / (max_velocity - min_velocity)],
                        [action])
    return activeTiles


def take_action(observation):
    if random.random() > epsilon:
        return np.argmax([qfunction(observation, action) for action in range(env.action_space.n)])
    else:
        return random.randint(0, 2)


# q(S,A,theta) = x(S,A).T thetas
def qfunction(observation, action):
    global thetas
    return np.matmul(features(observation, action), thetas)


# delta_q(S,A,theta) = x(S,A)
def delta(observation, action):
    return features(observation, action)


def features(observation, action):
    tileIndices = getActiveTiles(observation[0], observation[1], action)
    feature = [0] * maxtiles
    for tile_index in tileIndices:
        feature[tile_index] = 1
    return feature


def learn():
    global thetas
    global env
    cost2Go = []
    avg_reward = 0
    env = wrappers.Monitor(env, './semi-sarsa', force=True)
    for i_episode in range(2000):
        observation = env.reset()
        reward_total = 0
        action = take_action(observation)
        t = 0
        while True:
            t += 1
            newObservation, reward, done, info = env.step(action)
            env.render()
            reward_total += reward
            avg_reward += reward
            if done:
                change = alpha * (reward - qfunction(observation, action))
                thetas += change * np.array(delta(observation, action))
                break
            newAction = take_action(newObservation)
            qdash = qfunction(newObservation, newAction)
            q = qfunction(observation, action)
            change = alpha * (reward + gamma * qdash - q)
            thetas += change * np.array(delta(observation, action))
            observation = newObservation
            action = newAction
        if i_episode % 10 == 0 and i_episode != 0:
            cost2Go.append(-reward_total / 10)
            print("Episode# {} finished with avg. rewards {}".format(i_episode, t, avg_reward / 10))
            avg_reward = 0
        else:
            print("Episode# {} finished after {} timesteps with total rewards {}".format(i_episode, t, reward_total))
    env.close()

    plt.plot(cost2Go)
    plt.ylabel('cost to go')
    plt.show()

if __name__ == "__main__":
    learn()