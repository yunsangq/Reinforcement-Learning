import numpy as np
from tkinter import *

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
discount = 0.9

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        states.append([i, j])

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

qval = []
# policy
pval = dict({'L':0.9, 'U':0.033, 'R':0.033, 'D':0.033})
for i in range(0, WORLD_SIZE):
    qval.append([])
    for j in range(0, WORLD_SIZE):
        qval[i].append(dict({'L':0.0, 'U':0.0, 'R':0.0, 'D':0.0}))

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


# sub_3 1)
def find_states(pos):
    cnt = 0
    for i, j in states:
        if i == pos[0] and j == pos[1]:
            return cnt
        cnt += 1

# 25, 100
Policy = np.zeros((len(states), len(states)*len(actions)))
# 100, 25
transition = np.zeros((len(states)*len(actions), len(states)))
R = np.zeros((len(states)*len(actions), 1))
state_cnt = 0
pj_cnt = 0
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        for action in actions:
            Policy[state_cnt][pj_cnt] = 0.25
            R[pj_cnt] = actionReward[i][j][action]
            pj_cnt += 1
        state_cnt += 1

state_cnt = 0
tmp = 0
for i in range(len(states)*len(actions)):
    x,y = states[state_cnt]
    if tmp == 4:
        tmp = 0
        state_cnt += 1
    for action in actions:
        newPosition = nextState[x][y][action]
        state_idx = find_states(newPosition)
        transition[i][state_idx] += pval[action]
    tmp += 1

transition_p = np.dot(Policy, transition)
reward_p = np.dot(Policy, R)

v = np.dot(np.linalg.inv(np.identity(len(states)) - discount * transition_p), reward_p)

world = np.zeros((WORLD_SIZE, WORLD_SIZE))
v_cnt = 0
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        world[i][j] = v[v_cnt]
        v_cnt += 1

print('sub3_1)')
print('Random Policy')
print(world)


# sub_3 2)
def qmax(pos):
    tmp = []
    for action in actions:
        tmp.append(qval[pos[0]][pos[1]][action])
    return np.max(tmp)

world = np.zeros((WORLD_SIZE, WORLD_SIZE))
while True:
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(0, WORLD_SIZE):
        for j in range(0, WORLD_SIZE):
            values = []
            for action in actions:
                sigma_pv = 0.0
                for action_1 in actions:
                    newPos = nextState[i][j][action_1]
                    sigma_pv += pval[action_1] * world[newPos[0], newPos[1]]

                values.append(actionReward[i][j][action] + discount * sigma_pv)

                sigmaq = 0.0
                for action_2 in actions:
                    sigmaq += pval[action_2] * qmax(nextState[i][j][action_2])

                qval[i][j][action] = actionReward[i][j][action] + discount * sigmaq

            newWorld[i][j] = np.max(values)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('sub3_2)')
        print('Optimal Policy')
        print(newWorld)
        print('Optimal Policy')
        for i in range(WORLD_SIZE):
            print(", ".join('%.02f' % x for x in newWorld[i]))

        print('Action Value')
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                print('i: {}, j: {}, qval: {}'.format(i, j, qval[i][j]))
        master = Tk()
        w = Canvas(master, width=900, height=900)
        w.pack()
        i_cnt = 0
        for i in range(20, 770, 150):
            j_cnt = 0
            for j in range(20, 770, 150):
                w.create_rectangle(j, i, j+150, i+150)
                w.create_text(j+20, i+10, text=str(round(newWorld[i_cnt][j_cnt], 1)))
                w.create_line(j, i, j+150, i+150)
                w.create_line(j, i + 150, j + 150, i)
                # ['L', 'U', 'R', 'D']
                x1 = (j+(j+150))/2
                y1 = (i + (i + 150)) / 2

                w.create_text(x1 - 50, y1, text=str(round(qval[i_cnt][j_cnt]['L'], 1)))
                w.create_text(x1, y1 - 40, text=str(round(qval[i_cnt][j_cnt]['U'], 1)))
                w.create_text(x1 + 50, y1, text=str(round(qval[i_cnt][j_cnt]['R'], 1)))
                w.create_text(x1, y1 + 40, text=str(round(qval[i_cnt][j_cnt]['D'], 1)))

                if qval[i_cnt][j_cnt]['L'] >= newWorld[i_cnt, j_cnt]:
                    w.create_line(x1, y1, x1 - 30, y1, arrow=LAST)
                if qval[i_cnt][j_cnt]['U'] >= newWorld[i_cnt, j_cnt]:
                    w.create_line(x1, y1, x1, y1 - 30, arrow=LAST)
                if qval[i_cnt][j_cnt]['R'] >= newWorld[i_cnt, j_cnt]:
                    w.create_line(x1, y1, x1 + 30, y1, arrow=LAST)
                if qval[i_cnt][j_cnt]['D'] >= newWorld[i_cnt, j_cnt]:
                    w.create_line(x1, y1, x1, y1 + 30, arrow=LAST)
                j_cnt += 1
            i_cnt += 1

        mainloop()
        w.delete()
        break
    world = newWorld