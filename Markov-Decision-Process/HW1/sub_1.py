import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
discount = 0.9

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        states.append([i, j])

nextState = []
actionReward = []
for i in range(0, WORLD_SIZE):
    nextState.append([])
    actionReward.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
        reward = dict()
        if i == 0:
            next['U'] = [i, j]
            reward['U'] = -1.0
        else:
            next['U'] = [i - 1, j]
            reward['U'] = 0.0

        if i == WORLD_SIZE - 1:
            next['D'] = [i, j]
            reward['D'] = -1.0
        else:
            next['D'] = [i + 1, j]
            reward['D'] = 0.0

        if j == 0:
            next['L'] = [i, j]
            reward['L'] = -1.0
        else:
            next['L'] = [i, j - 1]
            reward['L'] = 0.0

        if j == WORLD_SIZE - 1:
            next['R'] = [i, j]
            reward['R'] = -1.0
        else:
            next['R'] = [i, j + 1]
            reward['R'] = 0.0

        if [i, j] == A_POS:
            next['L'] = next['R'] = next['D'] = next['U'] = A_PRIME_POS
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

        if [i, j] == B_POS:
            next['L'] = next['R'] = next['D'] = next['U'] = B_PRIME_POS
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

        nextState[i].append(next)
        actionReward[i].append(reward)


def find_states(pos):
    cnt = 0
    for i, j in states:
        if i == pos[0] and j == pos[1]:
            return cnt
        cnt += 1

# 25, 100
Policy = np.zeros((len(states), len(states)*len(actions)))
# 100, 25
transition = np.zeros((len(states), len(states)*len(actions)))
R = np.zeros((len(states)*len(actions), 1))
state_cnt = 0
pj_cnt = 0
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        for action in actions:
            Policy[state_cnt][pj_cnt] = 0.25

            newPosition = nextState[i][j][action]
            state_idx = find_states(newPosition)
            transition[state_idx][pj_cnt] = 1

            R[pj_cnt] = actionReward[i][j][action]
            pj_cnt += 1
        state_cnt += 1

transition = np.transpose(transition)

transition_p = np.dot(Policy, transition)
reward_p = np.dot(Policy, R)

v = np.dot(np.linalg.inv(np.identity(len(states)) - discount * transition_p), reward_p)

world = np.zeros((WORLD_SIZE, WORLD_SIZE))
v_cnt = 0
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        world[i][j] = v[v_cnt]
        v_cnt += 1

print('Random Policy')
print(world)