import numpy as np
from tkinter import *

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
discount = 0.9

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

qval = []
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
                newPosition = nextState[i][j][action]
                values.append(actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
                qval[i][j][action] = actionReward[i][j][action] + discount * qmax(newPosition)
            newWorld[i][j] = np.max(values)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        print(newWorld)
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