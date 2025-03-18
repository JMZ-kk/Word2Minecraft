import numpy as np
import heapq

import json
import os
import numpy as np
import pandas as pd
from numpy.ma.extras import average

from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from w2m_tools import *
import matplotlib.pyplot as plt
import math


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))  # (priority, (x, y))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return g_score[current]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] != 1:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return float('inf')  # No path found


def find_distances(grid):
    rows, cols = grid.shape
    start = None
    objectives = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 3:
                start = (r, c)
            elif grid[r, c] == 2:
                objectives.append((r, c))

    if start is None:
        raise ValueError("No start position found")

    distances = {obj: a_star(grid, start, obj) for obj in objectives}
    return distances

def fitness(main_map):
    if not isvalid(main_map):
        return -1
    distances = find_distances(main_map)
    avg_res = sum(distances.values()) / len(distances)
    if not math.isinf(avg_res):
        return avg_res
    else:
        return -1

def isvalid(main_map):
    pass

dir='word2world/examples/dataset'
model_list=['mini','example']
total_levels={'mini':0,'example':0}
success_levels={'mini':0,'example':0}
minmax_dict={'mini':[],'example':[]}
for filename in os.listdir(dir):
    if filename.startswith('ub'):
        for model_name in model_list:
            if model_name in filename:
                with open(f'{dir}/{filename}', 'r') as file:
                    data = json.load(file)
                    distances = find_distances(np.array(data['uw']))
                    avg_res = sum(distances.values()) / len(distances)
                    if not math.isinf(avg_res):
                        minmax_dict[model_name].append(avg_res)

mean_mini, std_mini = np.mean(minmax_dict['mini']), np.std(minmax_dict['mini'])
mean_turbo, std_turbo = np.mean(minmax_dict['example']), np.std(minmax_dict['example'])

models = ['Mini Model', 'Turbo Model']
plt.rcParams.update({'font.size': 18})
x = np.arange(len(models))  # x轴位置
width = 0.5  # 柱状图宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 画柱状图，带误差棒
bars = ax.bar(x, [mean_mini, mean_turbo], width, yerr=[std_mini, std_turbo], capsize=5, color=['r', 'orange'])

# 添加标签和标题
ax.set_xlabel('Models')
ax.set_ylabel('Distance')
ax.set_title('Average length of shortest paths to all objectives.')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_xlim(-0.5, len(models) - 0.5)
# 显示图表
plt.show()