import argparse
import json
import os
from collections import deque
import numpy as np
import openai
import pandas as pd
from sympy.unify.usympy import illegal
from configs import Config
from fixers import pad_rows_to_max_length
from solvers import find_characters
from utils import map_to_list, batch_similarity, extract_list, extract_dict
from gdpc import Editor,Block
from lookup import BLOCKS
from story_tools import *
block_data =  [item.replace('minecraft:', '') for item in BLOCKS]

# round_number='round_0'
# game_0='example_1'
# game_0_dir = os.path.join(f"word2world", "examples")
# with open(f'{game_0_dir}/{game_0}.json', 'r') as file_0:
#     data_0 = json.load(file_0)
#
# grid_str = data_0[round_number]["world"]
# grid_str = pad_rows_to_max_length(grid_str)
# grid_world_0 = map_to_list(grid_str)
# char_tile_mapping_0 = data_0[round_number]["tile_mapping"]  # real mapping: story
#
# story_gen_discriptions_0, story_gen_prompt = story_generator(grid_world_0, char_tile_mapping_0)
# print(story_gen_discriptions_0['choices'][0]['message']['content'])

path='word2world/blocks_none_scaling_1.json'
with open(path, 'r') as file_0:
     data = json.load(file_0)
     height=data['height']
     width=data['width']
     length=data['length']
     json_str = json.dumps(data['blocks'])
path_1='word2world/examples/new_example_1.json'
with open(path_1, 'r') as file_1:
     data_1 = json.load(file_1)
     objs=json.dumps(data_1['new_obj'])
# story_gen_discriptions, story_gen_prompt=story_generator_3D(json_str,height,width,length,objs, model=cfg.model)
image_path='word2world/images/example_1_tpd.png'
story_gen_discriptions, story_gen_prompt=generate_story_image(image_path,objs)

with open('word2world/story/obj_1.txt', 'w', encoding='utf-8') as file:
    file.write(story_gen_discriptions['choices'][0]['message']['content'])
print(story_gen_discriptions['choices'][0]['message']['content'])