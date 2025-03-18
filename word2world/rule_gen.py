import argparse
import json
import os
import numpy as np
import pandas as pd
from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from w2m_tools import *

game = "example_20"
game_dir = os.path.join(f"word2world", "examples")
with open(f'{game_dir}/{game}.json', 'r') as file:
    data = json.load(file)

df = pd.read_csv('word2world/items.csv', header=None)
df2 = pd.read_csv('word2world/enemy.csv', header=None)
df3 = pd.read_csv('word2world/tools.csv', header=None)
item_list = df[0].tolist()
enemy_list = df2[0].tolist()
tool_list = df3[0].tolist()

round_number = "round_0"
character_descriptions_dict = {}
story = data[round_number]['story']
grid_str = data[round_number]["world"]
grid_str = pad_rows_to_max_length(grid_str)
grid_world = map_to_list(grid_str)
char_tile_mapping = data[round_number]["tile_mapping"]
walkables = data[round_number]["walkable_tiles"]
objectives = data[round_number]["objectives"]
important_tiles = data[round_number]["important_tiles"]
goals = data[round_number]["goals"]
world_1st_layer = data[round_number]["world_1st_layer"]["world"]
world_1st_layer = pad_rows_to_max_length(world_1st_layer)
grid_1st_layer = map_to_list(world_1st_layer)
character_chars = find_characters(grid_str)
tiles_1st_layer = data[round_number]["world_1st_layer"]["tiles"]
player_pos = [character_chars['@'][0], character_chars['@'][1]]  # Starting position of the player
enemy_pos = [character_chars['#'][0], character_chars['#'][1]]  # Starting position of the enemy
walkable_tile_counts, default_walkable_tile = calculate_tile_count(grid_world, walkables)
tile_counts, default_tile = calculate_tile_count(grid_world)
sorted_items = sorted(tile_counts.items(), key=lambda x: x[1], reverse=True)
top_half = sorted_items[:len(tile_counts) // 2]
top_half_keys = {k for k, v in top_half}
story_prompt = f"Write a {cfg.story_paragraphs[0]}-{cfg.story_paragraphs[1]} paragraph story which has characters including the protagonist trying to achieve something and the antagonist wanting to stop the protagonist. The story should describe an environment(s) where the story is set up."
block_data = [item.replace('minecraft:', '') for item in BLOCKS]
description=list(objectives.keys())[0]

tile_dict, sub_tile_discriptions, sub_tile_prompt = (
    extract_subworld_tile(story, story_prompt,description, block_data))
sub_world,sub_world_discriptions,sub_world_prompt=(
    subworld_generation(story, story_prompt, sub_tile_discriptions, sub_tile_prompt, tile_dict))

sub_rule_prompt, sub_rule_discriptions=sub_rule_gen(story, story_prompt, sub_world,sub_world_discriptions, sub_world_prompt, sub_tile_discriptions, sub_tile_prompt)
ans=sub_cdblocks(story, story_prompt, sub_rule_prompt, sub_rule_discriptions,sub_world,sub_world_discriptions, sub_world_prompt, sub_tile_discriptions, sub_tile_prompt)
print()