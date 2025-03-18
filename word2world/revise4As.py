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

from word2world.map_scaling_2D import unwalkable_map

dir='word2world/examples/dataset'
model_list=['mini','example']

for filename in os.listdir(dir):
    for model_name in model_list:
        if filename.startswith(model_name):
            with open(f'{dir}/ub_{filename}', 'r') as file:
                data = json.load(file)
                round_number='round_0'
                grid_str = data[round_number]["world"]
                grid_str = pad_rows_to_max_length(grid_str)
                grid_world = map_to_list(grid_str)
                unwalkable_map=np.zeros(len(grid_world),len(grid_world[0]))
                for i in range(len(grid_world)):
                    for j in range(len(grid_world[0])):
                        if grid_world[i][j] == '@':
                            if data2['uw'][i][j]==2:
                                data2['uw'][i][j]=3
                            else:
                                print('bug')
                with open(f"{dir}/ub_{filename}", 'w') as f:
                    json.dump(data2, f)