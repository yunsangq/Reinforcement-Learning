from __future__ import print_function
import numpy as np
from tkinter import *
import time
import copy

WORLD_SIZE = 4
REWARD = -1.0

world = np.zeros((WORLD_SIZE, WORLD_SIZE))


# left, up, right, down
actions = ['L', 'U', 'R', 'D']

action_prob = []
for i in range(0, WORLD_SIZE):
    action_prob.append([])
    for j in range(0, WORLD_SIZE):
        action_prob[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

nextState = []
for i in range(0, WORLD_SIZE):
    nextState.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
        if i == 0:
            next['U'] = [i, j]
        else:
            next['U'] = [i - 1, j]

        if i == WORLD_SIZE - 1:
            next['D'] = [i, j]
        else:
            next['D'] = [i + 1, j]

        if j == 0:
            next['L'] = [i, j]
        else:
            next['L'] = [i, j - 1]

        if j == WORLD_SIZE - 1:
            next['R'] = [i, j]
        else:
            next['R'] = [i, j + 1]

        nextState[i].append(next)

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])


def policy_improvement():
    for i, j in states:
        max = -99999.0
        max_cnt = 0.0
        for action0 in actions:
            newPosition = nextState[i][j][action0]
            if max < world[newPosition[0], newPosition[1]]:
                max = world[newPosition[0], newPosition[1]]

        for action1 in actions:
            newPosition = nextState[i][j][action1]
            if np.abs(round(max, 1) - round(world[newPosition[0], newPosition[1]], 1)) == 0:
                max_cnt += 1.0

        for action2 in actions:
            newPosition = nextState[i][j][action2]
            if np.abs(round(max, 1) - round(world[newPosition[0], newPosition[1]], 1)) == 0:
                action_prob[i][j][action2] = 1.0/max_cnt
            else:
                action_prob[i][j][action2] = 0.0

idx = 0
save_world = []
save_action_prob = []
save_idx = []
start_time = time.time()
while True:
    new_action_prob = []
    for i in range(0, WORLD_SIZE):
        new_action_prob.append([])
        for j in range(0, WORLD_SIZE):
            new_action_prob[i].append(dict({'L': 0.0, 'U': 0.0, 'R': 0.0, 'D': 0.0}))

    if idx == 0:
        newlist = copy.copy(world)
        save_world.append(newlist)
        newactionlist = copy.deepcopy(action_prob)
        save_action_prob.append(newactionlist)
        save_idx.append(idx)

    delta = 0.0
    for i, j in states:
        newval = 0.0
        for action in actions:
            newPosition = nextState[i][j][action]
            newval += 0.25 * (REWARD + world[newPosition[0], newPosition[1]])
        delta = max(delta, np.abs(newval - world[i][j]))
        world[i][j] = newval

    if delta < 1e-4:
        print('Random Policy')
        print(world)
        newlist = copy.copy(world)
        save_world.append(newlist)
        policy_improvement()
        newactionlist = copy.deepcopy(action_prob)
        save_action_prob.append(newactionlist)
        save_idx.append(idx)
        break
    policy_improvement()
    idx += 1
    if idx == 1 or idx == 2 or idx == 3 or idx == 10:
        newlist = copy.copy(world)
        save_world.append(newlist)
        newactionlist = copy.deepcopy(action_prob)
        save_action_prob.append(newactionlist)
        save_idx.append(idx)


duration = time.time() - start_time
print('One array version time: %.3fsec' % duration)

master = Tk()
w = Canvas(master, width=1500, height=900)
w.pack()

w.create_text(130, 20, text='One array version time: %.3fsec' % duration)

for k in range(6):
    if k < 3:
        w.create_text(20, 120 + (k*230), text='k = ' + str(save_idx[k]))
        i_cnt = 0
        # y
        for i in range(40 + (k*230), 240 + (k*230), 50):
            j_cnt = 0
            # x
            for j in range(40, 240, 50):
                w.create_rectangle(j, i, j + 50, i + 50)
                w.create_text(j + 20, i + 10, text=str(round(save_world[k][i_cnt][j_cnt], 1)))

                j_cnt += 1
            i_cnt += 1

        i_cnt = 0
        # y
        for i in range(40 + (k*230), 240 + (k*230), 50):
            j_cnt = 0
            # x
            for j in range(40 + 240, 240 + 240, 50):
                w.create_rectangle(j, i, j + 50, i + 50)

                if (j_cnt == 0 and i_cnt == 0) or (j_cnt == 3 and i_cnt == 3):
                    pass
                else:
                    # ['L', 'U', 'R', 'D']
                    x1 = (j + (j + 50)) / 2
                    y1 = (i + (i + 50)) / 2

                    if save_action_prob[k][i_cnt][j_cnt]['L'] > 0:
                        w.create_line(x1, y1, x1 - 20, y1, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['U'] > 0:
                        w.create_line(x1, y1, x1, y1 - 20, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['R'] > 0:
                        w.create_line(x1, y1, x1 + 20, y1, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['D'] > 0:
                        w.create_line(x1, y1, x1, y1 + 20, arrow=LAST)

                j_cnt += 1
            i_cnt += 1
    else:
        w.create_text(40 + 240 + 230, 120 + ((k-3) * 230), text='k = ' + str(save_idx[k]))
        i_cnt = 0
        # y
        for i in range(40 + ((k-3) * 230), 240 + ((k-3) * 230), 50):
            j_cnt = 0
            # x
            for j in range(40 + 240 + 260, 240 + 240 + 260, 50):
                w.create_rectangle(j, i, j + 50, i + 50)
                w.create_text(j + 20, i + 10, text=str(round(save_world[k][i_cnt][j_cnt], 1)))

                j_cnt += 1
            i_cnt += 1

        i_cnt = 0
        # y
        for i in range(40 + ((k - 3) * 230), 240 + ((k - 3) * 230), 50):
            j_cnt = 0
            # x
            for j in range(40 + 240 + 260 + 240, 240 + 240 + 260 + 240, 50):
                w.create_rectangle(j, i, j + 50, i + 50)
                if (j_cnt == 0 and i_cnt == 0) or (j_cnt == 3 and i_cnt == 3):
                    pass
                else:
                    # ['L', 'U', 'R', 'D']
                    x1 = (j + (j + 50)) / 2
                    y1 = (i + (i + 50)) / 2

                    if save_action_prob[k][i_cnt][j_cnt]['L'] > 0:
                        w.create_line(x1, y1, x1 - 20, y1, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['U'] > 0:
                        w.create_line(x1, y1, x1, y1 - 20, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['R'] > 0:
                        w.create_line(x1, y1, x1 + 20, y1, arrow=LAST)
                    if save_action_prob[k][i_cnt][j_cnt]['D'] > 0:
                        w.create_line(x1, y1, x1, y1 + 20, arrow=LAST)

                j_cnt += 1
            i_cnt += 1
mainloop()
w.delete()