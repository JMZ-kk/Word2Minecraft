import argparse
import json
import os
from collections import deque
import numpy as np
import openai
import pandas as pd
from sympy.unify.usympy import illegal
from configs import Config
from fixers import pad_rows_to_max_length,remove_extra_special_chars
from solvers import find_characters,parse_grid
from utils import map_to_list, batch_similarity, extract_list, extract_dict,extract_between_ticks

from gdpc import Editor, Block
from lookup import BLOCKS, FILTERING
from small_buildings import *
from Objective import Objective
from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
import random
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
    # print(os.getcwd())
    # with open('word2world/examples/example_1.json', 'r') as file:
    #     data = json.load(file)


def extract_tile_set(char_tile_mapping, choices, item_check=False):
    cfg = Config()
    tile_list = list(char_tile_mapping.keys())
    tileset_prompt = (f'Given a list A: \n{tile_list}\n, '
                      f'and a list B: {choices},'
                      f'for each element in list A, please find the element in list B that has the closest meaning. '
                      f'Return only a Python Dictionary where each key is an element from list A, and '
                      f'its value is the most semantically similar element from list B. All value should be different')
    print(tileset_prompt)
    while True:
        tileset_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
            {"role": "user", "content": tileset_prompt}
        ], temperature=1)
        temp = extract_dict(tileset_discriptions['choices'][0]['message']['content'])
        if len(temp) == 0:
            continue
        break
    for key in temp.keys():
        if temp[key] not in choices:
            similarities = batch_similarity([key] * len(choices),
                                            choices, model_type='bert')
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            temp[key] = choices[max_index]
    tileset = {}
    for key, value in char_tile_mapping.items():
        tileset[value] = temp[key]
    print(tileset)
    print("\n")
    return tileset, tileset_discriptions, tileset_prompt


def extract_enemy(enemy_name, choices):
    cfg = Config()
    tile_list = list(char_tile_mapping.keys())
    tileset_prompt = (f'Given a enemy: \n{enemy_name}\n, '
                      f'and a list of monsters in Minecraft: {choices},'
                      f'please find the monster that has the closest meaning with the enemy. '
                      f'Return only the name of the monster that has the closest meaning.')
    print(tileset_prompt)
    tileset_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": tileset_prompt}
    ], temperature=1)
    temp = tileset_discriptions['choices'][0]['message']['content']
    if temp not in choices:
        similarities = batch_similarity([temp] * len(choices),
                                        choices, model_type='bert')
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        temp = choices[max_index]
    return temp


def extract_goals(objective_list, objective_types):
    cfg = Config()
    goal_prompt = (f'You will be given a list of objectives: \n{objective_list}\n '
                   f'and each objective\'s goal belongs to at least one of the '
                   f'following categories:\n{objective_types}\n'
                   f'If an objective\'s goal does not fit into any of these categories, '
                   f'classify it as \'Other\'.'
                   f'Only one onjective belongs to \'Defeat the enemy\'.'
                   f'Your task is to match each objective to its corresponding category and '
                   f'return a python dictionary where the key is objective and the value is the corresponding category (or \'Other\').')
    goal_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt}
    ], temperature=1)
    goal_dict = extract_dict(goal_discriptions['choices'][0]['message']['content'])
    return goal_prompt, goal_discriptions, goal_dict


def repair_goals(goal_prompt, goal_discriptions, old_list, cate_list):
    cfg = Config()
    repair_prompt = (
        f'For those objectives \n{old_list}\n do not fit into any of these categories, could you slightly change the'
        f'objectives to fit them into one of the category in \n{cate_list}\n?'
        f'Please return a python dict where the key is the old objective and '
        f'the value is a python list: [repaired objective,category].')
    repair_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": repair_prompt}
    ], temperature=1)
    repair_dict = extract_dict(repair_discriptions['choices'][0]['message']['content'])
    return repair_prompt, repair_discriptions, repair_dict


def extract_npc(goal_prompt, goal_discriptions, objective,tool_list):
    cfg = Config()
    npc_prompt = (f'You will be given a task: \n{objective}\n. What is '
                  f'the npc name in this task? What is his or her clothes type? '
                  f'and what is the hand item of the npc? The clothes type must be '
                  f'one of [golden,leather,diamond,iron]. The hand item should be one of '
                  f'\n{tool_list}\n'
                  f'Please return a python list with only 3 elements: The first element is the npc name,'
                  f'the second element is the clothes type and the third element is the hand item, '
                  f'like [\'Tom\', \'diamond\', \'golden_hoe\'].')
    print(npc_prompt)
    npc_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_prompt}
    ], temperature=1)
    npc_info = extract_list(npc_discriptions['choices'][0]['message']['content'])
    return npc_info, npc_prompt, npc_discriptions

def extract_npc_name(goal_prompt, goal_discriptions, objective_list):
    cfg = Config()
    npc_prompt = (f'You will be given a list of \'Interact with NPC\' objectives: \n{objective_list}\n '
                  f'return a python dict where the key is the objective and the value is the name of the npc.')
    print(npc_prompt)
    npc_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_prompt}
    ], temperature=1)
    npc_dict = extract_dict(npc_discriptions['choices'][0]['message']['content'])
    return npc_dict, npc_prompt, npc_discriptions


def extract_npc_appearance(goal_prompt, goal_discriptions, npc_prompt, npc_discriptions, objective_list):
    cfg = Config()
    npc_appearance_prompt = (f'For each npc in the objective list \n{objective_list}\n, '
                             f'choose one of [golden,leather,diamond,iron] '
                             f'to describe his or her clothes.'
                             f'return a python dict where the key is the task and the value is the clothes type.')
    npc_appearance_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_prompt},
        {"role": "assistant", "content": npc_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_appearance_prompt},
    ], temperature=1)
    npc_appearance_dict = extract_dict(npc_appearance_discriptions['choices'][0]['message']['content'])
    return npc_appearance_dict


def extract_tool(goal_prompt, goal_discriptions, npc_prompt, npc_discriptions, tool_list, objective_list):
    cfg = Config()
    tool_prompt = (
        f'You will be given a list of tools: \n{tool_list}\n and a list of objectives: \n{objective_list}\n. '
        f'Find one of them as hand item for the npc in each objective.'
        f'return a python dict where the key is the objective and the value is the npc\'s hand item.')
    tool_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_prompt},
        {"role": "assistant", "content": npc_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": tool_prompt}
    ], temperature=1)
    tool_dict = extract_dict(tool_discriptions['choices'][0]['message']['content'])
    return tool_dict


def extract_time(goal_prompt, goal_discriptions, repair_prompt, repair_discriptions, objective_list):
    cfg = Config()
    time_prompt = (f'You will be given a list of time quests: \n{objective_list}\n. '
                   f'For each objective, please find the time (in seconds).'
                   f'Please return a python dict where the key is the objective and the value is the time.')
    time_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": repair_prompt},
        {"role": "assistant", "content": repair_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": time_prompt}
    ], temperature=1)
    return extract_dict(time_discriptions['choices'][0]['message']['content'])


def extract_item(goal_prompt, goal_discriptions, objective_list, choices):
    cfg = Config()
    item_prompt = (f'You will be given a list of \'Collect items\' objectives: \n{objective_list}\n and '
                   f'a list of item choices: \n{choices}\n. '
                   f'For each objective\'s target item, please find the item choice that has the closest meaning.'
                   f'return a python dict where the key is the objective and the value is the item choice (not a list).')
    print(item_prompt)
    item_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": item_prompt}
    ], temperature=1)
    return extract_dict(item_discriptions['choices'][0]['message']['content'])


def pick_scaling_tiles(des2not, tile_map):
    cfg = Config()
    pick_building_prompt = (f'Given a 2D map \n{tile_map}\n, and a dictionary {des2not} where each key is a tile\'s '
                            f'description and each value is the notation for that tile in the map, '
                            f'identify which tile notations in the map need to be scaled?'
                            f'A tile is considered scaled if it should occupy more than one grid cell,'
                            f' such as a \'house\' tile. '
                            # f'Please avoid selecting adjacent tiles of the same type and frequent tiles.'
                            f'Please avoid scaling # and @'
                            f'Return a Python list of tiles, formatted as '
                            f'a list of scaled tile notations (e.g., `[\'a\',\'b\']`)')
    print(pick_building_prompt)
    pick_building_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": pick_building_prompt}
    ], temperature=1)
    building_tiles_list = extract_list(pick_building_discriptions['choices'][0]['message']['content'])
    print(building_tiles_list)
    print("\n")
    return building_tiles_list, pick_building_discriptions, pick_building_prompt


def extract_scaling_size(building_tiles_list, pick_building_discriptions, pick_building_prompt):
    cfg = Config()
    extract_size_prompt = ("Each scaled tile is a square."
                           "Assign each scaled tile in \n" + str(building_tiles_list) + "\n a side length, "
                                                                                        "formatted as a dict (e.g., {'a':'2','b':'3'}),"
                                                                                        " without any additional text or explanation.")
    print(pick_building_prompt)

    size_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": pick_building_prompt},
        {"role": "assistant", "content": pick_building_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": extract_size_prompt}
    ],
                                                     temperature=1)
    size_dict = extract_dict(size_discriptions['choices'][0]['message']['content'])
    print(size_dict)
    print("\n")
    return size_dict, size_discriptions, extract_size_prompt


def extract_subworld_tile(story, story_prompt, obj_des, choice):
    sub_tile_prompt = (f'Create an exhaustive list of 7-10 tiles in \n{choice}\n '
                       f'needed to create the environment of objetive: n{obj_des}\n.'
                       f'Please provide the output as a Python list of tiles')
    sub_tile_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": sub_tile_prompt}
    ], temperature=1)
    sub_tile_list = extract_list(sub_tile_discriptions['choices'][0]['message']['content'])
    import string
    sub_tile_dict = dict(zip(string.ascii_lowercase, sub_tile_list))
    return sub_tile_dict, sub_tile_discriptions, sub_tile_prompt

def subworld_generation(story, story_prompt,sub_tile_discriptions, sub_tile_prompt,sub_tile_dict):
    sub_world_prompt = (f'You are given a dictionary \n{sub_tile_dict}\n, the key is notation and'
                       f'the value is the tile. Please create a map for the objective by notations, '
                       f'which should contain at least 20 rows and 20 columns. Please return a python string list, where'
                       f'each element represents a line')
    sub_world_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": sub_tile_prompt},
        {"role": "assistant", "content": sub_tile_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_world_prompt}
    ], temperature=1)
    world_map_raw = extract_list(
        sub_world_discriptions['choices'][0]['message']['content'])
    max_length = max(len(line) for line in world_map_raw)
    padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in world_map_raw]
    return padded_lines,sub_world_discriptions,sub_world_prompt

def extract_submaze_tile(story, story_prompt, obj_des, choice):
    maze_tile_prompt = (f'Pick 2 tiles in the minecraft block list: \n{choice}\n '
                        f'to create the environment of objective: n{obj_des}\n.'
                        f'One as the path and one as the wall.'
                        f'Please provide the output as a Python list, '
                        f'where the first element is path tile and the second element is wall tile. '
                        f'Please make sure the 2 tiles are one of the minecraft block list.')
    maze_tile_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": maze_tile_prompt}
    ], temperature=1)
    maze_list = extract_list(maze_tile_discriptions['choices'][0]['message']['content'])
    return maze_list


def generate_maze_array():
    m = random.randint(7, 10)
    n = random.randint(7, 10)
    maze = Maze()
    maze.generator = BacktrackingGenerator(m, n)
    maze.generate()
    return maze.grid

def generate_building(story, story_prompt, building_description, choice, size):
    cfg = Config()
    generate_building_prompt = (f"Generate a {size}x{size} building with unlimited height that matches "
                                f"the following description: {building_description}. The building"
                                f" must only use blocks from the following list: \n"
                                f"{choice}\n. You should make sure the coherence between builidng and story."
                                f"Please provide the output as a Python list of tuples, "
                                f"where each tuple represents a block's coordinates "
                                f"and type in the format (x, y, z, 'block_name'). "
                                f"Assume the building's base starts at (0, 0, 0) and "
                                f"extends upward. Ensure the structure matches "
                                f"the description while keeping the block arrangement "
                                f"logical and visually consistent. Only return a Python list.")

    generate_building_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": generate_building_prompt}
    ],
                                                                  temperature=1)
    compo_list = extract_list(generate_building_discriptions['choices'][0]['message']['content'])
    print(compo_list)
    print("\n")
    return compo_list, generate_building_discriptions, generate_building_prompt


def modify_string(s, k, new_char):
    s_list = list(s)
    s_list[k] = new_char
    return ''.join(s_list)


def tile_evaluation(check, m, n, tile_counts, grid_world, tar):
    # 0-2: common tiles
    # 3: object
    # 4: scaling tile
    # 5: new house
    if check[m][n] == 3 or check[m][n] == 5:
        return -10000
    if check[m][n] == 4:
        if grid_world[m][n] == tar:
            return 200
        else:
            return -10000
    return tile_counts[grid_world[m][n]]


def block_evaluation(check, m, n, tile_counts, grid_world, tar, block_size):
    sum = 0
    for i in range(block_size):
        for j in range(block_size):
            try:
                sum += tile_evaluation(check, m + i, n + j, tile_counts, grid_world, tar)
            except:
                print(f'i:{m + i},j:{n + j}')
                raise
    return sum


def update_map(check, biggest_m, biggest_n, block_size, tar, grid_world):
    for i in range(block_size):
        for j in range(block_size):
            check[biggest_m + i][biggest_n + j] = 5
            grid_world[biggest_m + i] = modify_string(grid_world[biggest_m + i], biggest_n + j,
                                                      tar)


def find_nearest_character(matrix, char, x, y, walkable_tile_counts, occupied_positions):
    dis = 999
    res_x = -1
    res_y = -1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i, j) in occupied_positions:
                continue
            try:
                if (i-1, j) in occupied_positions:
                    continue
            except:
                pass
            try:
                if (i+1, j) in occupied_positions:
                    continue
            except:
                pass
            try:
                if (i, j-1) in occupied_positions:
                    continue
            except:
                pass
            try:
                if (i, j+1) in occupied_positions:
                    continue
            except:
                pass
            temp = abs(i - x) + abs(j - y)
            if matrix[i][j] == char:
                temp -= 999
            elif matrix[i][j] in walkable_tile_counts:
                temp -= walkable_tile_counts[matrix[i][j]]
            if temp < dis:
                dis = temp
                res_x = i
                res_y = j
    return res_x, res_y


def calculate_tile_count(world, choice=None):
    res = {}
    for row in world:
        for tile in row:
            if choice is None or tile in choice:
                if tile not in res:
                    res[tile] = 1
                else:
                    res[tile] += 1
    default_tile = max(res, key=res.get)
    return res, default_tile


def objectives_2_map(grid_world, objective_list, walkable_tile_counts, occupied_positions):
    for obj in objective_list:
        x, y = find_nearest_character(matrix=grid_world, char=obj.note, x=obj.coordinates[0],
                                      y=obj.coordinates[1], walkable_tile_counts=walkable_tile_counts,
                                      occupied_positions=occupied_positions)
        if x == -1:
            continue
        occupied_positions.append((x, y))
        obj.coordinates = (x, y)
        # grid_world[x] = modify_string(grid_world[x], y, '#')

def place_block(x,y,z,block):
    editor.placeBlock((x, y - 1, z), Block('minecraft:oak_wood'))
    editor.placeBlock((x, y, z), Block(block))

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

objective_list = []
objective_des = []
idx=1
level_id=random.randint(0,10000)
for des, val in objectives.items():
    objective_list.append(Objective(idx+level_id,des, val[0], (val[1], val[2])))
    objective_des.append(des)
    idx+=1

objective_types = ['Defeat the enemy', 'Chat with NPC', 'Navigate the maze to find an exit', 'Survive waves of enemies',
                   'Collect Items']
goal_prompt, goal_discriptions, goal_dict = extract_goals(objective_des, objective_types)
false_des = []
for obj in objective_list:
    if goal_dict[obj.description] == 'Other':
        false_des.append(obj.description)
    else:
        obj.goal = goal_dict[obj.description]
if len(false_des) > 0:
    repair_prompt, repair_discriptions, repair_dict = repair_goals(goal_prompt, goal_discriptions, false_des,
                                                                   objective_types)
    for obj in objective_list:
        for key in repair_dict.keys():
            if obj.description == key:
                obj.description = repair_dict[key][0]
                obj.goal = repair_dict[key][1]

for obj in objective_list:
    if obj.goal == 'Navigate the maze to find an exit':
        obj.map = generate_maze_array()
        tile_list = extract_submaze_tile(story, story_prompt, obj.description, block_data)
        obj.tile_dict={}
        obj.tile_dict['0']=tile_list[0]
        obj.tile_dict['1'] = tile_list[1]
    elif obj.goal=='Chat with NPC':
        obj.npc_info, npc_prompt, npc_discriptions=extract_npc(goal_prompt, goal_discriptions, obj.description,tool_list)
    else:
        obj.tile_dict, sub_tile_discriptions, sub_tile_prompt = extract_subworld_tile(story, story_prompt,
                                                                                      obj.description, block_data)
        obj.map,sub_world_discriptions,sub_world_prompt=subworld_generation(story, story_prompt,sub_tile_discriptions, sub_tile_prompt,obj.tile_dict)


occupied_positions = []
objectives_2_map(grid_world, objective_list, walkable_tile_counts, occupied_positions)

check = np.zeros((len(grid_world), len(grid_world[0])))
for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        if (i, j) in occupied_positions:
            check[i][j] = 3
        if f'''{grid_world[i][j]}''' not in walkables:
            for m in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    try:
                        if check[i + m][j + n] == 0:
                            check[i + m][j + n] = 2
                    except:
                        pass
            check[i][j] = 1
        elif grid_world[i][j] in ['#', '@']:
            for m in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    try:
                        if check[i + m][j + n] == 0:
                            check[i + m][j + n] = 2
                    except:
                        pass
            check[i][j] = 3

non_special_char_map = {k: v for k, v in data[round_number]["tile_mapping"].items()
                        if v not in ['#', '@']}
bottom_half_des2not = {k: v for k, v in non_special_char_map.items() if v not in top_half_keys}

enemy = [k for k, v in data[round_number]["tile_mapping"].items() if v == '#'][0]

# ------------------------------------------scaling problem--------------------------------------
scaling_tiles_list, pick_scaling_discriptions, pick_scaling_prompt = pick_scaling_tiles(bottom_half_des2not, grid_world)
size_dict, size_discriptions, extract_size_prompt = extract_scaling_size(scaling_tiles_list, pick_scaling_discriptions,
                                                                         pick_scaling_prompt)
tileset, tileset_discriptions, tileset_prompt = extract_tile_set(non_special_char_map, block_data)
enemy_block = extract_enemy(enemy, enemy_list)
compo_dict = {}

for notation, size in size_dict.items():
    for des, sub_note in data[round_number]["tile_mapping"].items():
        if notation == sub_note:
            compo_list, generate_building_discriptions, generate_building_prompt = (
                generate_building(story, story_prompt, des, building_blocks, size))
            compo_dict[notation] = compo_list
            break

for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        if grid_world[i][j] in scaling_tiles_list:
            check[i][j] = 4

for building in scaling_tiles_list:
    for i in range(len(grid_world)):
        for j in range(len(grid_world[0])):
            if grid_world[i][j] == building:
                block_size = int(size_dict[grid_world[i][j]])
                tar = grid_world[i][j]
                biggest_m = -1
                biggest_n = -1
                biggest = 0
                for m in [i - block_size + 1, i]:
                    for n in [j - block_size + 1, j]:
                        try:
                            tsum = block_evaluation(check, m, n, tile_counts, grid_world, tar, block_size)
                            if tsum > biggest:
                                biggest = tsum
                                biggest_n = n
                                biggest_m = m
                        except Exception as e:
                            check[i][j] = 5
                            grid_world[i] = modify_string(grid_world[i], j, default_walkable_tile)
                            print(e)
                if biggest_m > -1:
                    update_map(check, biggest_m, biggest_n, block_size, tar, grid_world)
print(grid_world)
# ------------------------------------------scaling problem--------------------------------------


# Minecraft map geenration

editor = Editor(buffering=False)

base_x = 0 * (len(grid_world) + 3)
base_z = 0 * (len(grid_world[0]) + 3)
base_y = -20
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
        if (x, z) in occupied_positions:
            for obj in objective_list:
                if obj.coordinates == (x, z):
                    start_tag = 'chat_with_' + str(obj.id)
                    end_tag = 'chat_with_' + str(obj.id) + '_done'
                    if obj.goal == 'Chat with NPC':
                        cloth = obj.npc_info[1]
                        command = (
                                'summon armor_stand ' + str(base_x + x) + ' ' + str(base_y + 1) + ' ' + str(
                            base_z + z) + ' {'
                                          'CustomName:\'{\\\"text\\\":\\\"' + obj.npc_info[
                                    0] + '\\\"}\',CustomNameVisible:1b'
                                         ',ArmorItems:[{id:\\\"minecraft:' + cloth + '_boots\\\",Count:1},'
                                                                                     '{id:\\\"minecraft:' + cloth + '_leggings\\\",Count:1},'
                                                                                                                    '{id:\\\"minecraft:' + cloth + '_chestplate\\\",Count:1},'
                                                                                                                                                   '{id:\\\"minecraft:' + cloth + '_helmet\\\",Count:1}],'
                                                                                                                                                                                  'HandItems:[{id:\\\"minecraft:' +
                                obj.npc_info[2] + '\\\",Count:1},{}]}'
                        )
                        print(command)
                        command1 = (
                                'execute as @a[tag=!' + start_tag + '] at @s if entity @e[type=armor_stand, name=\\\"' +
                                obj.npc_info[0] + '\\\", distance=..2] run tag @s add ' + start_tag
                        )

                        command2 = (
                                'execute as @a[tag=' + start_tag + ',tag=!' + end_tag + '] run tellraw @s {\\\"text\\\":\\\"' + obj.description + '\\\",\\\"color\\\":\\\"green\\\"}'
                        )

                        command3 = (
                                'execute as @a[tag=' + start_tag + ',tag=!' + end_tag + '] run tag @s add ' + end_tag)
                        tag_list.append(end_tag)
                        editor.placeBlock((base_x, 60 + base_y + summon_block_num, base_z),
                                          Block('minecraft:command_block',
                                                data=f'{{"Command": "{command}","auto": 1}}'))
                        summon_block_num += 1
                        editor.placeBlock((base_x + x, base_y, base_z + z),
                                          Block('minecraft:repeating_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command1}","auto": 1}}'))
                        editor.placeBlock((base_x + x, base_y - 1, base_z + z),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command2}","auto": 1}}'))
                        editor.placeBlock((base_x + x, base_y - 2, base_z + z),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command3}","auto": 1}}'))
                        continue
                    sum_x_len += 2
                    for m in range(len(obj.map)):
                        for n in range(len(obj.map[0])):
                            tx = sum_x_len + m
                            ty = base_y
                            tz = base_z + n
                            if obj.goal == 'Navigate the maze to find an exit':
                                path_tile = obj.tile_dict['0']
                                wall_tile = obj.tile_dict['1']
                                if obj.map[m][n] == 0:
                                    place_block(tx, ty, tz, 'minecraft:' + path_tile)
                                else:
                                    place_block(tx, ty, tz, 'minecraft:' + wall_tile)
                                    editor.placeBlock((tx, ty + 1, tz),
                                                      Block('minecraft:' + wall_tile))
                                    editor.placeBlock((tx, ty + 2, tz),
                                                      Block('minecraft:barrier'))
                            else:
                                try:
                                    place_block(tx, ty, tz, 'minecraft:' + obj.tile_dict[obj.map[m][n]])
                                except:
                                    place_block(tx, ty, tz, 'minecraft:' + tileset.get(default_walkable_tile))
                                    print('bug:'+str(obj.map[m][n]))
                    maze_end_x=sum_x_len + len(obj.map)-2
                    maze_end_z= base_z + len(obj.map[0])-2

                    if obj.goal == 'Survive waves of enemies':
                        timer='t'+str(obj.id)
                        tag='rs'+str(obj.id)
                        time_command_2='execute if score global Tick matches 18 run scoreboard players remove @a[scores={'+timer+'=1..}] '+timer+' 1'
                        time_command_3='title @a[scores={'+timer+'=1..}] actionbar {\\\"text\\\":\\\"Countdown: \\\",\\\"color\\\":\\\"yellow\\\",\\\"extra\\\":[{\\\"score\\\":{\\\"name\\\":\\\"@a\\\",\\\"objective\\\":\\\"'+timer+'\\\"}}]}'
                        time_command_4='execute as @a[scores={'+timer+'=0}] run title @a title {\\\"text\\\":\\\"Enemy is coming!\\\",\\\"color\\\":\\\"red\\\"}'
                        time_command_5='execute as @a[scores={'+timer+'=..0}] run scoreboard players reset @a '+timer
                        time_command_6 = 'execute as @a[scores={'+timer+'=0},tag=!'+tag+'] run summon zombie '+str(maze_end_x-2)+' '+ str(base_y+2)+' '+ str(maze_end_z-2)
                        command_7='tag @a add '+tag
                        editor.placeBlock((maze_end_x, base_y - 1, maze_end_z-1), Block('minecraft:command_block',
                                    data='{"Command": "scoreboard objectives add '+timer+' dummy","auto": 1})'))
                        editor.placeBlock((maze_end_x, base_y - 2, maze_end_z-1), Block('minecraft:command_block',
                                    data='{"Command": "scoreboard players set @a '+timer+' 10","auto": 1})'))
                        editor.placeBlock((maze_end_x, base_y - 3, maze_end_z - 1),
                                          Block('minecraft:repeating_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{time_command_2}","auto": 1}}'))
                        editor.placeBlock((maze_end_x, base_y - 4, maze_end_z-1),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{time_command_3}","auto": 1}}'))
                        editor.placeBlock((maze_end_x, base_y - 5, maze_end_z-1),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{time_command_4}","auto": 1}}'))
                        editor.placeBlock((maze_end_x, base_y - 6, maze_end_z - 1),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{time_command_5}","auto": 1}}'))

                        editor.placeBlock((maze_end_x - 1, base_y - 2, maze_end_z),
                                          Block('minecraft:repeating_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{time_command_6}","auto": 1}}'))
                        editor.placeBlock((maze_end_x - 1, base_y - 3, maze_end_z),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command_7}","auto": 1}}'))
#-------------------------------tp-----------------------
                    editor.placeBlock((base_x + x, base_y, base_z + z), Block('minecraft:command_block',
                                                                           data='{"Command": "tp @p ' + str(
                                                                               sum_x_len + 1) + ' ' + str(
                                                                               base_y + 4) + ' ' + str(
                                                                               base_z + 1) + '"}'))
                    editor.placeBlock((base_x + x, base_y+1, base_z + z), Block('minecraft:stone_pressure_plate'))
#-------------------------------tp back-----------------------
                    editor.placeBlock((maze_end_x, base_y, maze_end_z), Block('minecraft:command_block', states={'facing': 'down'},
                                                                           data='{"Command": "tp @p ' + str(
                                                                               base_x + player_pos[1]) + ' ' + str(
                                                                               base_y + 4) + ' ' + str(
                                                                               base_z + player_pos[0]) + '"}'))
                    editor.placeBlock((maze_end_x, base_y+1, maze_end_z), Block('minecraft:stone_pressure_plate'))
                    command2=f'setblock {base_x + x} {base_y} {base_z + z} minecraft:'+ tileset.get(default_walkable_tile)
                    command3=f'setblock {base_x + x} {base_y+1} {base_z + z} minecraft:air'
                    editor.placeBlock((maze_end_x, base_y - 1, maze_end_z),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command2}","auto": 1}}'))
                    editor.placeBlock((maze_end_x, base_y - 2, maze_end_z),
                                          Block('minecraft:chain_command_block', states={'facing': 'down'},
                                                data=f'{{"Command": "{command3}","auto": 1}}'))

                    title_command = (
                            'summon armor_stand ' + str(base_x + x) + ' ' + str(base_y + 1) + ' ' + str(
                        base_z + z) + ' {'
                                      'CustomName:\'{\\\"text\\\":\\\"' + obj.description + '\\\"}\','
                                                                                            'CustomNameVisible:1b,NoGravity:1b,Invisible:1b}'
                    )
                    editor.placeBlock((base_x, 60 + base_y + summon_block_num, base_z), Block('minecraft:command_block',
                                                 data=f'{{"Command": "{title_command}","auto": 1}}'))
                    summon_block_num += 1
                    if obj.map is not None:
                        sum_x_len += len(obj.map)
                    break
        else:
            if grid_world[x][z] in scaling_tiles_list and finished_scaling[x][z] == 0:
                size = int(size_dict[grid_world[x][z]])
                for i in range(size):
                    for j in range(size):
                        try:
                            finished_scaling[x + i][z + j] = 1
                        except:
                            pass
                for tuple in compo_dict[grid_world[x][z]]:
                    editor.placeBlock((base_x + x + tuple[0], base_y + tuple[1], base_z + z + tuple[2]),
                                      Block('minecraft:' + str(tuple[3])))

            if z == player_pos[0] and x == player_pos[1]:
                editor.placeBlock((base_x + x, base_y, base_z + z), Block('minecraft:command_block',
                                                                          data='{"Command": "kill @e[type=minecraft:enderman]"}'))
            else:
                try:
                    tile = tileset.get(base_map[x][z])
                    block = 'minecraft:' + tile
                    if 'leaves' in tile:
                        editor.placeBlock((base_x + x, base_y - 1, base_z + z), Block('minecraft:oak_wood'))
                        editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))
                    elif block in FILTERING:
                        editor.placeBlock((base_x + x, base_y - 1, base_z + z), Block(block))
                    elif base_map[x][z] in walkables:
                        editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))
                    else:
                        for h_y in range(3):
                            editor.placeBlock((base_x + x, base_y + h_y, base_z + z), Block(block))
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

time_command='scoreboard players add global Tick 1'
time_command_1='execute if score global Tick matches 20 run scoreboard players set global Tick 0'

editor.placeBlock((base_x,-50,base_z+1),
                  Block('minecraft:repeating_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{time_command}","auto": 1}}'))
editor.placeBlock((base_x,-51,base_z+1),
                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{time_command_1}","auto": 1}}'))
# editor.placeBlock((base_x,-53,base_z+1),
#                   Block('minecraft:chain_command_block', states={'facing': 'down'},
#                         data=f'{{"Command": "{time_command_3}","auto": 1}}'))
# editor.placeBlock((base_x,-54,base_z+1),
#                   Block('minecraft:chain_command_block', states={'facing': 'down'},
#                         data=f'{{"Command": "{time_command_4}","auto": 1}}'))
#
# start_tag='summon_monster'
# end_tag='summon_monster_done'
# z = enemy_pos[0]
# x = enemy_pos[1]
#
# base_condi='tag=!'+start_tag+',tag=!'+end_tag
# for tag in tag_list:
#     base_condi+=',tag='+tag
#
# command1 = ('execute as @a['+base_condi+'] run tag @s add '+start_tag)
# command2 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tellraw @s '
#             '{\\\"text\\\":\\\"Defeat the enemy\\\",\\\"color\\\":\\\"green\\\"}')
# try:
#     command3 = (f'execute as @a[tag={start_tag},tag=!{end_tag}] run summon '
#                 f'minecraft:{enemy_block} {base_x + x} {base_y + 2} {base_z + z}')
#     command4 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] if entity @e[type=minecraft:'+enemy_block+', distance=..5] run tag @s add '+end_tag)
#     print(enemy_block)
# except:
#     command3 = (f'execute as @a[tag={start_tag},tag=!{end_tag}] run summon '
#                 f'minecraft:zombie {base_x + x} {base_y+2} {base_z + z}')
#     command4 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] if entity @e[type=minecraft:zombie, distance=..5] run tag @s add '+end_tag)
# editor.placeBlock((base_x + x, base_y, base_z + z),
#                     Block('minecraft:repeating_command_block', states={'facing': 'down'},
#         data=f'{{"Command": "{command1}","auto": 1}}'))
# editor.placeBlock((base_x + x, base_y-1, base_z + z),
#                     Block('minecraft:chain_command_block', states={'facing': 'down'},
#         data=f'{{"Command": "{command2}","auto": 1}}'))
# editor.placeBlock((base_x + x, base_y-2, base_z + z),
#                     Block('minecraft:chain_command_block', states={'facing': 'down'},
#         data=f'{{"Command": "{command3}","auto": 1}}'))
# editor.placeBlock((base_x + x, base_y-3, base_z + z),
#                     Block('minecraft:chain_command_block', states={'facing': 'down'},
#         data=f'{{"Command": "{command4}","auto": 1}}'))
