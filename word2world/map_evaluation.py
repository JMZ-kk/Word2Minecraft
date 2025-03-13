import argparse
import json
import os
import numpy as np
import pandas as pd
from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from w2m_tools import *
from story_tools import *
from utils import map_to_list, find_most_similar_images
import numpy as np
import matplotlib.pyplot as plt

round_number = "round_0"
game_list=['20','20','20','20','20']
res_list={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}
for game_0 in game_list:
    game_0_dir = os.path.join(f"word2world", "examples")
    with open(f'{game_0_dir}/example_{game_0}.json', 'r') as file_0:
        data_0 = json.load(file_0)
    old_story=data_0[round_number]["story"]
    char_tile_mapping_0 = data_0[round_number]["tile_mapping"]
    grid_str = data_0[round_number]["world"]
    grid_str = pad_rows_to_max_length(grid_str)
    grid_world_0 = map_to_list(grid_str)#old map

    path_1 = f'word2world/examples/new_example_{game_0}.json'
    with open(path_1, 'r') as file_1:
        data_1 = json.load(file_1)
        objs = json.dumps(data_1['new_obj'])

    game_1 = f'new_{game_0}'
    game_1_dir = os.path.join(f"word2world", "examples")
    with open(f'word2world/story/none_obj_{game_0}.txt', 'r') as file_1:
        story_1 = file_1.read()
    with open(f'word2world/story/obj_{game_0}.txt', 'r') as file_2:
        story_2 = file_2.read()
    random_story=random_story_generator()


    # char_tile_mapping_2, _ = (
    #     find_most_similar_images(char_tile_mapping_0, cfg.tile_data_dir))#bert: Oryx Design Lab
    # char_tile_mapping_3, _ = (
    #     find_most_similar_images(char_tile_mapping_0, cfg.tile_data_dir,isMC=True))#bert: Minecraft
    #
    # char_tile_mapping_2 = {v: k for k, v in char_tile_mapping_2.items()}
    # char_tile_mapping_3 = {v: k for k, v in char_tile_mapping_3.items()}
    #bert: Oryx Design Lab

    story_gen_discriptions_0,story_gen_prompt=story_generator(grid_world_0, char_tile_mapping_0)
    story_gen_discriptions_1, _ = story_generator_obj(grid_world_0, char_tile_mapping_0,objs, model=cfg.model)
    story_gen_discriptions_2, _ = story_generator_obj_only(objs, model=cfg.model)
    # #baseline
    #
    # story_gen_discriptions_1,_=story_generator(grid_world_1, char_tile_mapping_0)#upbound
    # story_gen_discriptions_2,_=story_generator(grid_world_1, char_tile_mapping_1)
    # story_gen_discriptions_3,_=story_generator(grid_world_1, char_tile_mapping_2)

    story_dict= {'0':random_story,'1':story_1,'2':story_2,
                 '3':story_gen_discriptions_0['choices'][0]['message']['content'],
                 '4':story_gen_discriptions_1['choices'][0]['message']['content'],
                 '5':story_gen_discriptions_2['choices'][0]['message']['content'],}

    res,compare_prompt=story_list_comparator_embd(old_story, story_dict)
    for key in res:
        res_list[key].append(res[key])

    print(game_0,end=':\n')
    print(res)
categories = list(res_list.keys())
values = list(res_list.values())

x = np.arange(len(values[0]))
width = 0.08
fig, ax = plt.subplots()
label_list=['random','3D','3D+obj','2D map + real tiles','2D map + real tiles + obj','obj only']
for i, (key, val) in enumerate(res_list.items()):
    ax.bar(x + i * width, val, width, label=f'{label_list[i]}')

ax.set_xticks(x + width / 2)
ax.set_xticklabels([f'Map {i+1}' for i in range(len(values[0]))])

ax.set_ylabel('Score')
ax.set_title('Coherence with old story')
ax.legend()

plt.show()
plt.savefig('score.png')
