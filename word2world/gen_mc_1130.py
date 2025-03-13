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
from gdpc import Editor
from lookup import BLOCKS
block_data =  [item.replace('minecraft:', '') for item in BLOCKS]

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

def extract_tile_set(char_tile_mapping,choices,item_check=False):
    cfg=Config()
    tile_list=list(char_tile_mapping.keys())
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
        if len(temp)==0:
            continue
        break
    for key in temp.keys():
        if temp[key] not in choices:
            similarities = batch_similarity([key] * len(choices),
                                            choices, model_type='bert')
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            temp[key] = choices[max_index]
    tileset={}
    for key, value in char_tile_mapping.items():
        tileset[value] = temp[key]
    print(tileset)
    print("\n")
    return tileset, tileset_discriptions,tileset_prompt

def extract_enemy(enemy_name,choices):
    cfg=Config()
    tile_list=list(char_tile_mapping.keys())
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

def extract_goals(objective_list,objective_types):
    cfg=Config()
    goal_prompt = (f'You will be given a list of objectives: \n{objective_list}\n '
                      f'and each objective belongs to one of the following categories:\n{objective_types}\n'
                      f'If an objective does not fit into any of these categories, classify it as \'Other\'.'
                      f'Your task is to match each objective to its corresponding category and '
                      f'return a python dictionary where: The key is the objective and the value is the name of the'
                      f' corresponding category (or "Other" if it does not belong to any of the five categories).')
    print(goal_prompt)
    goal_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt}
    ], temperature=1)
    goal_dict = extract_dict(goal_discriptions['choices'][0]['message']['content'])

    return goal_prompt, goal_discriptions, goal_dict


def extract_npc(goal_prompt, goal_discriptions, objective_list):
    cfg = Config()
    npc_prompt = (f'You will be given a list of \'Interact with NPC\' objectives: \n{objective_list}\n '
                   f'return a python dict where the key is the objective and the value is the npc.')
    print(npc_prompt)
    npc_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": npc_prompt}
    ], temperature=1)
    npc_dict = extract_dict(npc_discriptions['choices'][0]['message']['content'])
    return npc_dict

def extract_item(goal_prompt, goal_discriptions, objective_list,choices):
    cfg = Config()
    item_prompt = (f'You will be given a list of \'Collect items\' objectives: \n{objective_list}\n and '
                  f'a list of item choices: \n{choices}\n. '
                  f'For each objective\'s target item, please find the item choice that has the closest meaning.' 
                   f'return a python dict where the key is the objective and the value is the item choice.')
    print(item_prompt)
    item_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": item_prompt}
    ], temperature=1)
    return extract_dict(item_discriptions['choices'][0]['message']['content'])

# def scaling_map_generation(pick_building_discriptions, pick_building_prompt, important_list):
#     cfg=Config()
#     map_generation_prompt = ("Try to generate a new map by scaling those tiles need to be scaled. "
#                              "You can expand the size of the map if necessary, but try to generate a new map "
#                              "similar to the original map. "
#                              "Return a Python list to represent the map, formatted as (e.g., `[['a','b'],['c','d']]`)'")
#     print(pick_building_prompt)
#     size_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
#         {"role": "user", "content": pick_building_prompt},
#         {"role": "assistant", "content": pick_building_discriptions['choices'][0]['message']['content']},
#         {"role": "user", "content": map_generation_prompt}
#     ],
#                                                               temperature=1)
#     print(size_discriptions['choices'][0]['message']['content'])
#     new_map_list = extract_list(size_discriptions['choices'][0]['message']['content'])
#     print(new_map_list)
#     print("\n")
#     return new_map_list, size_discriptions,map_generation_prompt

def pick_scaling_tiles(tiles_1st_layer, tile_map):
    cfg=Config()
    pick_building_prompt = (f'Given a 2D map \n{tile_map}\n, and a dictionary {tiles_1st_layer} where each key is a tile\'s '
                            f'description and each value is the notation for that tile in the map, '
                            f'identify which tile notations in the map need to be scaled?'
                            f'A tile is considered scaled if it should occupy more than one grid cell,'
                            f' such as a \'house\' tile. Please avoid selecting adjacent tiles of the same type.'
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
    return building_tiles_list, pick_building_discriptions,pick_building_prompt

def extract_building_size(building_tiles_list, pick_building_discriptions, pick_building_prompt):
    cfg=Config()
    extract_size_prompt = ("Each scaled tile is a square."
                           "Assign each scaled tile in \n"+str(building_tiles_list)+"\n a side length, "
                           "formatted as a dict (e.g., {'a':'2','b':'3'}),"
                            " without any additional text or explanation.")
    print(pick_building_prompt)
    story_prompt = f"Write a {cfg.story_paragraphs[0]}-{cfg.story_paragraphs[1]} paragraph story which has characters including the protagonist trying to achieve something and the antagonist wanting to stop the protagonist. The story should describe an environment(s) where the story is set up."

    size_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": pick_building_prompt},
        {"role": "assistant", "content": pick_building_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": extract_size_prompt}
    ],
                                                              temperature=1)
    size_dict = extract_dict(size_discriptions['choices'][0]['message']['content'])
    print(size_dict)
    print("\n")
    return size_dict, size_discriptions,extract_size_prompt

def modify_string(s, k, new_char):
    s_list = list(s)
    s_list[k] = new_char
    return ''.join(s_list)


def find_nearest_character(matrix, char, x, y,npc_positions=None):
    dis=999
    res_x=-1
    res_y=-1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == char:
                if npc_positions is not None:
                    isIllegal=False
                    for tp in npc_positions:
                        if tp[0] == x and tp[1] == y:
                            isIllegal=True
                            break
                    if isIllegal:
                        continue
                temp=abs(i-x)+abs(j-y)
                if temp < dis:
                    dis=temp
                    res_x=i
                    res_y=j
    return res_x,res_y

#collect message of original map

round_number = "round_0"
character_descriptions_dict = {}
gen_story = data[round_number]["story"]
grid_str = data[round_number]["world"]
grid_str = pad_rows_to_max_length(grid_str)
grid_world = map_to_list(grid_str)
char_tile_mapping = data[round_number]["tile_mapping"]
walkables = data[round_number]["walkable_tiles"]
objectives = data[round_number]["objectives"]
important_tiles = data[round_number]["important_tiles"]
interactive_object_tiles = data[round_number]["interactive_object_tiles"]
goals = data[round_number]["goals"]
world_1st_layer = data[round_number]["world_1st_layer"]["world"]
world_1st_layer = pad_rows_to_max_length(world_1st_layer)
grid_1st_layer = map_to_list(world_1st_layer)
character_chars = find_characters(grid_str)
player_pos = [character_chars['@'][0], character_chars['@'][1]]  # Starting position of the player
enemy_pos = [character_chars['#'][0], character_chars['#'][1]]  # Starting position of the enemy
walkable_tile_counts = {}
for row in world_1st_layer:
    for tile in row:
        if tile in walkables:
            if tile not in walkable_tile_counts:
                walkable_tile_counts[tile] = 1
            else:
                walkable_tile_counts[tile] += 1
default_walkable_tile = max(walkable_tile_counts, key=walkable_tile_counts.get)


objective_list=objectives.keys()

objective_types=['Collect items', 'Interact with NPC', 'Avoid traps', 'Defeat enemies']
goal_prompt, goal_discriptions, goal_dict=extract_goals(objective_list,objective_types)
goal_dict_2={}
npc_positions={}
item_positions={}

for t in objective_types:
    goal_dict_2[t]=[]
goal_dict_2['Other']=[]
for key,value in goal_dict.items():
    goal_dict_2[value].append(key)

npc_dict=extract_npc(goal_prompt, goal_discriptions, goal_dict_2['Interact with NPC'])
item_dict = extract_item(goal_prompt, goal_discriptions, goal_dict_2['Collect items'],block_data)
npc_dict_reverse={}
item_dict_reverse={}
for task in goal_dict_2['Interact with NPC']:
    x,y=find_nearest_character(grid_world,objectives[task][0],objectives[task][1],objectives[task][2])
    if x==-1:
        continue
    grid_world[x] = modify_string(grid_world[x], y,'#')
    npc_positions[npc_dict[task]]=(x,y)
    npc_dict_reverse[npc_dict[task]]=task

for task in goal_dict_2['Collect items']:
    x,y=find_nearest_character(grid_world,objectives[task][0],objectives[task][1],objectives[task][2],npc_positions)
    if x==-1:
        continue
    grid_world[x] = modify_string(grid_world[x], y, '#')
    item_positions[item_dict[task]]=(x,y)
    item_dict_reverse[item_dict[task]]=task

tile_counts = {}
for row in world_1st_layer:
    for tile in row:
        if tile not in tile_counts:
            tile_counts[tile] = 1
        else:
            tile_counts[tile] += 1

default_tile = max(tile_counts, key=tile_counts.get)

sorted_items = sorted(tile_counts.items(), key=lambda x: x[1], reverse=True)
top_half = sorted_items[:len(tile_counts) // 2]
top_half_keys = {k for k, v in top_half}

check=np.zeros((len(grid_world), len(grid_world[0])))
for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        if f'''{grid_world[i][j]}''' not in walkables:
            for m in [-1,0,1]:
                for n in [-1,0,1]:
                    try:
                        if check[i+m][j+n]==0:
                            check[i+m][j+n]=2
                    except:
                        pass
            check[i][j]=1
        elif f'''{grid_world[i][j]}''' in interactive_object_tiles or grid_world[i][j] in ['#','@']:
            for m in [-1,0,1]:
                for n in [-1,0,1]:
                    try:
                        if check[i+m][j+n]==0:
                            check[i+m][j+n]=2
                    except:
                        pass
            check[i][j]=3

tiles_1st_layer = data[round_number]["world_1st_layer"]["tiles"]
# print(world_1st_layer)
# print()
# print(tiles_1st_layer)
# exit(0)


story=data[round_number]['story']
non_object_dict={k: v for k, v in data[round_number]["tile_mapping"].items()
                 if v not in interactive_object_tiles}
object_dict={k: v for k, v in data[round_number]["tile_mapping"].items()
                 if v in interactive_object_tiles}
enemy=[k for k, v in data[round_number]["tile_mapping"].items() if v == '#'][0]


building_tiles_list, pick_scaling_discriptions, pick_scaling_prompt=pick_scaling_tiles(non_object_dict,world_1st_layer)
size_dict, size_discriptions,extract_size_prompt=extract_building_size(building_tiles_list, pick_scaling_discriptions, pick_scaling_prompt)

df = pd.read_csv('word2world/items.csv', header=None)
df2 = pd.read_csv('word2world/enemy.csv', header=None)
item_list = df[0].tolist()
enemy_list = df2[0].tolist()
item_tileset, item_tileset_discriptions,item_tileset_prompt=extract_tile_set(object_dict,item_list)
tileset, tileset_discriptions,tileset_prompt=extract_tile_set(non_object_dict,block_data)
enemy_block=extract_enemy(enemy,enemy_list)

# building_tiles_list=['u']
# size_dict={'u':3}

# building_tiles_list=['u','i','v']
# size_dict={'i': '3', 'u': '4', 'v': '2'}
# building_tiles_list = sorted(building_tiles_list, key=lambda x: tile_counts[x])

for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        if grid_world[i][j] in building_tiles_list:
            check[i][j]=4

def tile_evaluation(check,m,n,tile_counts,grid_world,tar):

    # 0-2: common tiles
    # 3: object
    # 4: house
    # 5: new house
    if check[m][n] == 3 or check[m][n] == 5:
        return -10000
    if check[m][n] == 4:
        if grid_world[m][n] == tar:
            return 200
        else:
            return -10000
    return tile_counts[grid_world[m][n]]

def block_evaluation(check,m,n,tile_counts,grid_world,tar,block_size):
    sum=0
    for i in range(block_size):
        for j in range(block_size):
            try:
                sum+=tile_evaluation(check,m+i,n+j,tile_counts,grid_world,tar)
            except:
                print(f'i:{m+i},j:{n+j}')
                raise
    return sum

def update_map(check,biggest_m,biggest_n,block_size,tar):
    for i in range(block_size):
        for j in range(block_size):
            check[biggest_m+i][biggest_n+j] = 5
            grid_world[biggest_m + i] = modify_string(grid_world[biggest_m + i], biggest_n+j,
                                                          tar)

for building in building_tiles_list:
    for i in range(len(grid_world)):
        for j in range(len(grid_world[0])):
            if grid_world[i][j]==building:
                block_size=int(size_dict[grid_world[i][j]])
                tar=grid_world[i][j]
                biggest_m=-1
                biggest_n=-1
                biggest=0
                for m in [i-block_size+1,i]:
                    for n in [j-block_size+1,j]:
                        try:
                            tsum=block_evaluation(check,m,n,tile_counts,grid_world,tar,block_size)
                            if tsum>biggest:
                                biggest=tsum
                                biggest_n=n
                                biggest_m=m
                        except Exception as e:
                            check[i][j] = 5
                            grid_world[i] = modify_string(grid_world[i], j, default_walkable_tile)
                            print(e)
                if biggest_m>-1:
                    update_map(check, biggest_m, biggest_n, block_size, tar)
print(grid_world)


#Minecraft map geenration

editor = Editor(buffering=False)

from lookup import FILTERING
from small_buildings import *

scaling_size = 1
base_x = 2 * scaling_size *(len(grid_world) + 3)
base_z = -1 * scaling_size *(len(grid_world[0]) + 3)
base_y = -30
print(base_x)
print(base_z)

mid_padding=int(scaling_size/2)
print('mid_padding: ',end='')
print(mid_padding)

for x in range(len(grid_world)*scaling_size):
    for z in range(len(grid_world[0])*scaling_size):
        editor.placeBlock((base_x + x-1, base_y-1, base_z-1 + z), Block('minecraft:dirt'))
        editor.placeBlock((base_x + x-1, base_y - 2, base_z-1 + z), Block('minecraft:white_wool'))
base_map=grid_world
# base_map=grid_1st_layer

chasing_mode=False
summon_block_num=0
for x in range(len(base_map)):
    for z in range(len(base_map[x])):
        tx = x * scaling_size
        tz = z * scaling_size
        npc_found=False
        item_found=False
        for npc_name, value in npc_positions.items():
            task= npc_dict_reverse[npc_name]
            npc_name_str=npc_name.replace(' ', '_')
            start_tag='talking_with_'+npc_name_str
            end_tag='talk_with_'+npc_name_str+'_done'
            if value[0]==x and value[1]==z:
                command = (
                    'summon armor_stand '+str(base_x+x)+' '+str(base_y+1)+' '+str(base_z+z)+' {CustomName:\'{\\\"text\\\":\\\"'+npc_name_str+'\\\"}\',CustomNameVisible:1b'
                    ',ArmorItems:[{id:\\\"minecraft:golden_boots\\\",Count:1},{id:\\\"minecraft:golden_leggings\\\",Count:1},{id:\\\"minecraft:golden_chestplate\\\",Count:1},{id:\\\"minecraft:golden_helmet\\\",Count:1}],HandItems:[{id:\\\"minecraft:golden_pickaxe\\\",Count:1},{}],Pose:{RightArm:[-30f,0f,0f]}}'
                )

                command1 = (
                    'execute as @a[tag=!'+start_tag+'] at @s if entity @e[type=armor_stand, name=\\\"'+npc_name_str+'\\\", distance=..2] run tag @s add '+start_tag
                )

                command2 = (
                    'execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tellraw @s {\\\"text\\\":\\\"'+task+'\\\",\\\"color\\\":\\\"green\\\"}'
                )

                command3 = (
                    'execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tag @s add '+end_tag
                )

                editor.placeBlock((base_x, 60+base_y+summon_block_num, base_z), Block('minecraft:command_block',
                        data=f'{{"Command": "{command}","auto": 1}}'))
                summon_block_num+=1
                editor.placeBlock((base_x + tx, base_y, base_z + tz),
                                  Block('minecraft:repeating_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command1}","auto": 1}}'))
                editor.placeBlock((base_x + tx, base_y-1, base_z + tz),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command2}","auto": 1}}'))
                editor.placeBlock((base_x + tx, base_y-2, base_z + tz),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command3}","auto": 1}}'))
                npc_found = True
                break
        if npc_found:
            continue
        for item_name, value in item_positions.items():
            task= item_dict_reverse[item_name]
            item_name_str=item_name.replace(' ', '_')
            start_tag='talking_with_'+item_name_str
            end_tag='talk_with_'+item_name_str+'_done'
            if value[0]==x and value[1]==z:
                chest = Block("chest", data='{Items: [{Slot: 13b, id: ' + str(item_name) + ', Count: 1b}]}')
                editor.placeBlock((base_x + tx + mid_padding, base_y + 1, base_z + tz + mid_padding), chest)
                item_found = True
                break
        if item_found:
            continue
        if z == player_pos[0] and x == player_pos[1]:
            editor.placeBlock((base_x + tx + mid_padding, base_y, base_z + tz + mid_padding), Block('minecraft:command_block',
                    data='{"Command": "kill @e[type=minecraft:enderman]"}'))
        elif z == enemy_pos[0] and x == enemy_pos[1]:
            if chasing_mode:
                editor.placeBlock((base_x + tx + mid_padding, base_y, base_z + tz + mid_padding), Block('minecraft:command_block',
                    data='{"Command": "summon minecraft:enderman '+str(base_x + tx)+' '+str(base_y)+' '+str(base_z + tz)+'"}'))
                editor.placeBlock((base_x + tx + mid_padding, base_y+1, base_z + tz + mid_padding),Block('minecraft:oak_pressure_plate'))
                editor.placeBlock((base_x + tx + mid_padding, base_y-1, base_z + tz + mid_padding), Block('minecraft:command_block',
                    data='{"Command": "setblock '+str(base_x + player_pos[1]*scaling_size)+' '+str(base_y+1)+' '+str(base_z + player_pos[0]*scaling_size)+' minecraft:oak_pressure_plate"}'))
            else:
                block = enemy_block
                editor.placeBlock((base_x + tx + mid_padding, base_y + 1, base_z + tz + mid_padding), Block('minecraft:red_concrete'))
                chest = Block("chest", data='{Items: [{Slot: 13b, id: ' + str(block) + ', Count: 1b}]}')
                editor.placeBlock((base_x + tx + mid_padding, base_y + 2, base_z + tz + mid_padding), chest)

        elif base_map[x][z] in interactive_object_tiles:
            block = item_tileset.get(base_map[x][z]).lower().replace(" ", "_")
            if block=='bow' or block=='arrow':
                chest = Block("chest", data='{Items: [{Slot: 12b, id: bow, Count: 1b},{Slot: 13b, id: arrow, Count: 30b}]}')
            else:
                chest = Block("chest", data='{Items: [{Slot: 13b, id: ' + str(block) + ', Count: 1b}]}')
            editor.placeBlock((base_x + tx + mid_padding, base_y, base_z + tz + mid_padding), chest)
        else:
            tile=tileset.get(base_map[x][z])
            block = 'minecraft:' + tile
            try:
                if 'leaves' in tile:
                    editor.placeBlock((base_x + tx + mid_padding, base_y-1, base_z + tz + mid_padding), Block('minecraft:oak_wood'))
                    for a in range(scaling_size):
                        for b in range(scaling_size):
                            editor.placeBlock((base_x + tx + a, base_y, base_z + tz + b), Block(block))
                elif block in FILTERING:
                    for a in range(scaling_size):
                        for b in range(scaling_size):
                            editor.placeBlock((base_x + tx + a, base_y-1, base_z + tz + b), Block(block))
                elif base_map[x][z] in walkables:
                    if 'sapling' in tile:
                        editor.placeBlock((base_x + tx + mid_padding, base_y, base_z + tz + mid_padding), Block(block))
                    else:
                        for a in range(scaling_size):
                            for b in range(scaling_size):
                                editor.placeBlock((base_x + tx + a, base_y, base_z + tz + b), Block(block))
                else:
                    for a in range(scaling_size):
                        for b in range(scaling_size):
                            for h_y in range(3):
                                editor.placeBlock((base_x + tx + a, base_y+h_y, base_z + tz + b), Block(block))
            except:
                print(block, 'at', x, ',', z)

editor.placeBlock((base_x,-63,base_z-1), Block('minecraft:command_block',
        data='{"Command": "tp @p '+str(base_x+player_pos[1]*scaling_size)+' '+str(base_y+2)+' '+str(base_z+player_pos[0]*scaling_size)+'","auto": 1}'))

editor.placeBlock((base_x,-63,base_z), Block('minecraft:command_block',
        data='{"Command": "clear @p","auto": 1})'))

# editor.placeBlock((0,-63,2), Block('minecraft:chain_command_block',
#         data='{"Command": "gamemode survival","auto": 1})'))