import argparse
import json
import os
import numpy as np
import pandas as pd
from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from w2m_tools import *

block_data = [item.replace('minecraft:', '') for item in BLOCKS]
building_blocks = block_data.remove('water')
cfg = Config()
# Define constants
TILE_SIZE = 16
CAMERA_WIDTH = 20
CAMERA_HEIGHT = 16
INFO_PANEL_HEIGHT = 4  # Number of tiles for the info panel

parser = argparse.ArgumentParser(description="Process game inputs")
parser.add_argument('--game_path', type=str,
                    help="A path to JSON file of your game. Derfaults to 'word2world\examples\example_1.json'")
args = parser.parse_args()

if args.game_path:
    if not os.path.exists(args.game_path):
        raise ValueError(f"{args.game_path} does not exist. Please provide an existing path.")
    with open(args.game_path, 'r') as file:
        data = json.load(file)
else:
    game = "example_1"
    game_dir = os.path.join(f"word2world", "examples")
    with open(f'{game_dir}/{game}.json', 'r') as file:
        data = json.load(file)



# collect message of original map
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

non_special_char_map = {k: v for k, v in data[round_number]["tile_mapping"].items()
                        if v not in ['#', '@']}
bottom_half_des2not = {k: v for k, v in non_special_char_map.items() if v not in top_half_keys}

tileset, tileset_discriptions, tileset_prompt = extract_tile_set(non_special_char_map, block_data)

editor = Editor(buffering=False)

base_x = 5 * (len(grid_world) + 3)
base_z = 1 * (len(grid_world[0]) + 3)
base_y = -10
print(base_x)
print(base_z)

for x in range(len(grid_world)):
    for z in range(len(grid_world[0])):
        editor.placeBlock((base_x + x, base_y - 1, base_z + z), Block('minecraft:dirt'))
        editor.placeBlock((base_x + x, base_y - 2, base_z + z), Block('minecraft:white_wool'))
base_map = grid_world
# base_map=grid_1st_layer

chasing_mode = False
summon_block_num = 0
sum_x_len = base_x + len(grid_world)
tag_list = []
finished_scaling = np.zeros((len(grid_world), len(grid_world[0])))
for x in range(len(base_map)):
    for z in range(len(base_map[x])):
        try:
            tile = tileset.get(base_map[x][z])
            block = 'minecraft:' + tile
            if base_map[x][z] not in walkables and block not in FILTERING:
                for h_y in range(3):
                    editor.placeBlock((base_x + x, base_y + h_y, base_z + z), Block(block))
            else:
                place_block(editor,base_x + x, base_y, base_z + z, tile,block_data)
        except:
                # print(tile, 'at', x, ',', z)
                print(base_map[x][z])
                block = 'minecraft:' + tileset.get(default_walkable_tile)
                editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))


editor.placeBlock((base_x,-63,base_z-1), Block('minecraft:command_block',
        data='{"Command": "tp @p '+str(base_x+player_pos[1])+' '+str(base_y+4)+' '+str(base_z+player_pos[0])+'","auto": 1}'))

editor.placeBlock((base_x,-63,base_z), Block('minecraft:command_block',
        data='{"Command": "clear @p","auto": 1})'))

editor.placeBlock((base_x,-63,base_z-2), Block('minecraft:command_block',
        data='{"Command": "scoreboard objectives add Tick dummy","auto": 1})'))