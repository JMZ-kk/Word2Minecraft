from collections import deque
import argparse
import json
import os
import numpy as np
import pandas as pd
from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from w2m_tools import *
import matplotlib.pyplot as plt

def is_all_objective_accessible(grid):
    rows, cols = len(grid), len(grid[0])
    important_positions = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2}

    if not important_positions:
        return True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def bfs(start):
        queue = deque([start])
        visited = {start}

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] != 1:
                        queue.append((nr, nc))
                        visited.add((nr, nc))

        return visited
    start = next(iter(important_positions))
    reachable = bfs(start)
    return important_positions.issubset(reachable)

dir='word2world/examples/dataset'
model_list=['mini','example']
total_levels={'mini':0,'example':0}
success_levels={'mini':0,'example':0}
valid_area_ratio={'mini':[], 'example':[]}
total_area_ratio={'mini':[], 'example':[]}
print(os.listdir(dir))
for filename in os.listdir(dir):
    if filename.startswith('u'):
        for model_name in model_list:
            if model_name in filename:
                total_levels[model_name] += 1
                with open(f'{dir}/{filename}', 'r') as file:
                    data = json.load(file)
                    count_1 = np.sum(np.array(data['uw']) == 1)
                    count_2 = np.sum(np.array(data['uw']) == 2)
                    total_area_ratio[model_name].append((count_1 + count_2) / (len(data['uw']) * len(data['uw'][0])))
                    if is_all_objective_accessible(data['uw']):
                        valid_area_ratio[model_name].append((count_1 + count_2) / (len(data['uw']) * len(data['uw'][0])))
                        success_levels[model_name]+=1
                    else:
                        valid_area_ratio[model_name].append(0)
mean_mini, std_mini = np.mean(valid_area_ratio['mini']), np.std(valid_area_ratio['mini'])
mean_turbo, std_turbo = np.mean(valid_area_ratio['example']), np.std(valid_area_ratio['example'])
t_mean_mini, t_std_mini = np.mean(total_area_ratio['mini']), np.std(total_area_ratio['mini'])
t_mean_turbo, t_std_turbo = np.mean(total_area_ratio['example']), np.std(total_area_ratio['example'])

print(success_levels['mini']/total_levels['mini'])
print(success_levels['example']/total_levels['example'])
# 示例数据（Mini和Turbo模型的unwalkable tile占比）
models = ['Mini Model', 'Turbo Model']
plt.rcParams.update({'font.size': 18})
x = np.arange(len(models))  # x轴位置
width = 0.5  # 柱状图宽度

fig, ax = plt.subplots(figsize=(6, 6))

# 画柱状图，带误差棒
bars = ax.bar(x, [mean_mini, mean_turbo], width, yerr=[std_mini, std_turbo], capsize=5, color=['r', 'orange'])

# 添加标签和标题
ax.set_xlabel('Models')
ax.set_ylabel('Unwalkable Tile Ratio')
ax.set_title('Comparison of Unwalkable Tile Ratio')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_xlim(-0.5, len(models) - 0.5)
# 显示图表
plt.show()