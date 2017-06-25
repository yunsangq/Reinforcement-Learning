from __future__ import print_function
import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt
# goal
GOAL = 100

# all states, including state 0 and state 100
states = np.arange(GOAL + 1)

# probability of head (pi)
headProb = 0.4

# optimal policy
policy = np.zeros(GOAL + 1)

# action value
actionReward = []
actionValue = []
for i in range(0, GOAL + 1):
    actionValue.append([])
    actionReward.append([])
    for j in range(0, min(i, GOAL - i) + 1):
        actionValue[i].append(0)
        if i+j == GOAL:
            actionReward[i].append(1)
        else:
            actionReward[i].append(0)


def qmax(_state):
    tmp = []
    _actions = np.arange(min(_state, GOAL - _state) + 1)
    for _action in _actions:
        tmp.append(actionValue[_state][_action])
    return np.max(tmp)

# value iteration
while True:
    delta = 0.0
    for state in states[1:GOAL]:
        # get possilbe actions for current state
        actions = np.arange(min(state, GOAL - state) + 1)
        newval = []
        for action in actions:
            newval.append(actionReward[state][action] +
                          headProb * qmax(state+action) + (1-headProb) * qmax(state-action))
        delta += np.sum(np.abs(np.array(actionValue[state]) - np.array(newval)))
        # update value
        actionValue[state] = newval
    if delta < 1e-9:
        print(delta)
        break

# calculate the optimal policy
for state in states[1:GOAL]:
    actions = np.arange(min(state, GOAL - state) + 1)
    actionReturns = []
    for action in actions:
        actionReturns.append(actionValue[state][action])
    # due to tie and precision, can't reproduce the optimal policy in book
    policy[state] = actions[argmax(actionReturns)]

# figure 4.3
plt.figure(1)
plt.scatter(states, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()