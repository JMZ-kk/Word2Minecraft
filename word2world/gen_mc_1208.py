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
from lookup import BLOCKS,FILTERING
from small_buildings import *
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
    # with open(f'{game_dir}/{game}.json', 'r') as file:
    #     data = json.load(file)
    # print(os.getcwd())
    with open('word2world/examples/dataset/gpt4o_37999_0.json', 'r') as file:
        data = json.load(file)

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

def pick_scaling_tiles(tiles_1st_layer, tile_map):
    cfg=Config()
    pick_building_prompt = (f'Given a 2D map \n{tile_map}\n, and a dictionary {tiles_1st_layer} where each key is a tile\'s '
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
    return building_tiles_list, pick_building_discriptions,pick_building_prompt

def extract_scaling_size(building_tiles_list, pick_building_discriptions, pick_building_prompt):
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

def generate_building(building_description,choice,size):
    cfg=Config()
    generate_building_prompt = (f"Generate a {size}x{size} building with unlimited height that matches "
                                f"the following description: {building_description}. The building"
                                f" must only use blocks from the following list: \n"
                                f"{choice}\n "
                                f"Please provide the output as a Python list of tuples, "
                                f"where each tuple represents a block's coordinates "
                                f"and type in the format (x, y, z, 'block_name'). "
                                f"Assume the building's base starts at (0, 0, 0) and "
                                f"extends upward. Ensure the structure matches "
                                f"the description while keeping the block arrangement "
                                f"logical and visually consistent. Only return a Python list.")

    generate_building_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": generate_building_prompt}
    ],
                                                              temperature=1)
    compo_list = extract_list(generate_building_discriptions['choices'][0]['message']['content'])
    print(compo_list)
    print("\n")
    return compo_list, generate_building_discriptions,generate_building_prompt

def modify_string(s, k, new_char):
    s_list = list(s)
    s_list[k] = new_char
    return ''.join(s_list)

def tile_evaluation(check,m,n,tile_counts,grid_world,tar):

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

def update_map(check,biggest_m,biggest_n,block_size,tar,grid_world):
    for i in range(block_size):
        for j in range(block_size):
            check[biggest_m+i][biggest_n+j] = 5
            grid_world[biggest_m + i] = modify_string(grid_world[biggest_m + i], biggest_n+j,
                                                          tar)

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
df = pd.read_csv('word2world/items.csv', header=None)
df2 = pd.read_csv('word2world/enemy.csv', header=None)
item_list = df[0].tolist()
enemy_list = df2[0].tolist()

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
goals = data[round_number]["goals"]
world_1st_layer = data[round_number]["world_1st_layer"]["world"]
world_1st_layer = pad_rows_to_max_length(world_1st_layer)
grid_1st_layer = map_to_list(world_1st_layer)
character_chars = find_characters(grid_str)
tiles_1st_layer = data[round_number]["world_1st_layer"]["tiles"]
player_pos = [character_chars['@'][0], character_chars['@'][1]]  # Starting position of the player
enemy_pos = [character_chars['#'][0], character_chars['#'][1]]  # Starting position of the enemy
walkable_tile_counts = {}
for row in grid_world:
    for tile in row:
        if tile in walkables:
            if tile not in walkable_tile_counts:
                walkable_tile_counts[tile] = 1
            else:
                walkable_tile_counts[tile] += 1
default_walkable_tile = max(walkable_tile_counts, key=walkable_tile_counts.get)

cor2note = {}
for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        temp = grid_world[i][j]
        s = f'({i},{j})'
        cor2note[s] = temp


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
trap_positions_list=[]
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

for task in goal_dict_2['Avoid traps']:
    x,y=find_nearest_character(grid_world,objectives[task][0],objectives[task][1],objectives[task][2],npc_positions)
    if x==-1:
        continue
    grid_world[x] = modify_string(grid_world[x], y, '#')
    trap_positions_list.append((x,y))

tile_counts = {}
for row in grid_world:
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
        elif grid_world[i][j] in ['#','@']:
            for m in [-1,0,1]:
                for n in [-1,0,1]:
                    try:
                        if check[i+m][j+n]==0:
                            check[i+m][j+n]=2
                    except:
                        pass
            check[i][j]=3

story=data[round_number]['story']
non_special_char_map={k: v for k, v in data[round_number]["tile_mapping"].items()
                      if v not in ['#','@']}
enemy=[k for k, v in data[round_number]["tile_mapping"].items() if v == '#'][0]


scaling_tiles_list, pick_scaling_discriptions, pick_scaling_prompt=pick_scaling_tiles(non_special_char_map, cor2note)
size_dict, size_discriptions,extract_size_prompt=extract_scaling_size(scaling_tiles_list, pick_scaling_discriptions, pick_scaling_prompt)

tileset, tileset_discriptions,tileset_prompt=extract_tile_set(non_special_char_map, block_data)
enemy_block=extract_enemy(enemy,enemy_list)

scaling_tiles_list=['n', 'G']
size_dict={'n':3,'G':4}
compo_dict={}
for notation,size in size_dict.items():
    for des,sub_note in data[round_number]["tile_mapping"].items():
        if notation==sub_note:
            compo_list, generate_building_discriptions, generate_building_prompt = (
                generate_building(des, block_data, size))
            compo_dict[notation]=compo_list
            break


for i in range(len(grid_world)):
    for j in range(len(grid_world[0])):
        if grid_world[i][j] in scaling_tiles_list:
            check[i][j]=4

for building in scaling_tiles_list:
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
                    update_map(check, biggest_m, biggest_n, block_size, tar,grid_world)
print(grid_world)


#Minecraft map geenration

editor = Editor(buffering=False)

base_x = 0 * (len(grid_world) + 3)
base_z = 0 * (len(grid_world[0]) + 3)
base_y = -30
print(base_x)
print(base_z)

for x in range(len(grid_world)):
    for z in range(len(grid_world[0])):
        editor.placeBlock((base_x + x-1, base_y-1, base_z-1 + z), Block('minecraft:dirt'))
        editor.placeBlock((base_x + x-1, base_y - 2, base_z-1 + z), Block('minecraft:white_wool'))
base_map=grid_world
# base_map=grid_1st_layer

chasing_mode=False
summon_block_num=0
tag_list=[]
finished_scaling=np.zeros((len(grid_world),len(grid_world[0])))
for x in range(len(base_map)):
    for z in range(len(base_map[x])):
        npc_found=False
        item_found=False
        for npc_name, value in npc_positions.items():
            task= npc_dict_reverse[npc_name]
            npc_name_str=npc_name.replace(' ', '_')
            start_tag='talk_with_'+npc_name_str
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

                command3 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tag @s add '+end_tag)
                tag_list.append(end_tag)
                editor.placeBlock((base_x, 60+base_y+summon_block_num, base_z), Block('minecraft:command_block',
                        data=f'{{"Command": "{command}","auto": 1}}'))
                summon_block_num+=1
                editor.placeBlock((base_x + x, base_y, base_z + z),
                                  Block('minecraft:repeating_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command1}","auto": 1}}'))
                editor.placeBlock((base_x + x, base_y-1, base_z + z),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command2}","auto": 1}}'))
                editor.placeBlock((base_x + x, base_y-2, base_z + z),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command3}","auto": 1}}'))
                npc_found = True
                break
        if npc_found:
            continue
        for item_name, value in item_positions.items():
            task= item_dict_reverse[item_name]
            item_name_str=item_name.replace(' ', '_')
            start_tag='collect_'+item_name_str
            end_tag='collect_'+item_name_str+'_done'
            if value[0]==x and value[1]==z:
                chest = Block("chest", data='{Items: [{Slot: 13b, id: ' + str(item_name) + ', Count: 1b}]}')
                editor.placeBlock((base_x + x, base_y + 1, base_z + z), chest)
                command1 = (
                    'execute as @a[tag=!'+start_tag+'] at @s if entity @a[nbt={Inventory:[{id:\\\"minecraft:' + str(item_name) + '\\\"}]}] run tag @s add '+start_tag
                )

                command2 = (
                    'execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tellraw @s {\\\"text\\\":\\\"'+task+'\\\",\\\"color\\\":\\\"green\\\"}'
                )

                command3 = (
                    'execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tag @s add '+end_tag
                )
                tag_list.append(end_tag)
                editor.placeBlock((base_x + x, base_y, base_z + z),
                                  Block('minecraft:repeating_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command1}","auto": 1}}'))
                editor.placeBlock((base_x + x, base_y-1, base_z + z),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command2}","auto": 1}}'))
                editor.placeBlock((base_x + x, base_y-2, base_z + z),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command3}","auto": 1}}'))

                item_found = True
                break
        if item_found:
            continue
        if grid_world[x][z] in scaling_tiles_list and finished_scaling[x][z]==0:
            size=int(size_dict[grid_world[x][z]])
            for i in range(size):
                for j in range(size):
                    try:
                        finished_scaling[x+i][z+j]=1
                    except:
                        pass
            for tuple in compo_dict[grid_world[x][z]]:
                editor.placeBlock((base_x+x+tuple[0], base_y+tuple[1], base_z+z+tuple[2]), Block('minecraft:' + str(tuple[3])))

        if z == player_pos[0] and x == player_pos[1]:
            editor.placeBlock((base_x + x, base_y, base_z + z), Block('minecraft:command_block',
                    data='{"Command": "kill @e[type=minecraft:enderman]"}'))
        else:
            try:
                tile = tileset.get(base_map[x][z])
                block = 'minecraft:' + tile
                if 'leaves' in tile:
                    editor.placeBlock((base_x + x, base_y-1, base_z + z), Block('minecraft:oak_wood'))
                    editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))
                elif block in FILTERING:
                    editor.placeBlock((base_x + x, base_y-1, base_z + z), Block(block))
                elif base_map[x][z] in walkables:
                    editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))
                else:
                    for h_y in range(3):
                        editor.placeBlock((base_x + x, base_y+h_y, base_z + z), Block(block))
            except:
                # print(tile, 'at', x, ',', z)
                print(base_map[x][z])
                block = 'minecraft:' +tileset.get(default_walkable_tile)
                editor.placeBlock((base_x + x, base_y, base_z + z), Block(block))
        trap_num=0
        for tuple in trap_positions_list:
            if tuple[0] == x and tuple[1] == z:
                end_tag='trap_'+str(trap_num)
                print(f'trip at{base_x+x},{base_z+z}')
                command1 = (
                        f'execute as @a[x={base_x+x},y={base_y+1},z={base_z+z},tag=!{end_tag},distance=..3] run effect give @s nausea 10 1'
                )
                command2 = (
                        'execute as @a[tag=!'+end_tag+'] run tag @s add '+end_tag
                )
                editor.placeBlock((base_x + x, base_y-1, base_z + z),
                                  Block('minecraft:repeating_command_block', states={'facing': 'down'},
                                        data=f'{{"Command": "{command1}","auto": 1}}'))
                editor.placeBlock((base_x + x, base_y-2, base_z + z),
                                  Block('minecraft:chain_command_block', states={'facing': 'down'},
                        data=f'{{"Command": "{command2}","auto": 1}}'))
                trap_num+=1



editor.placeBlock((base_x,-63,base_z-1), Block('minecraft:command_block',
        data='{"Command": "tp @p '+str(base_x+player_pos[1])+' '+str(base_y+4)+' '+str(base_z+player_pos[0])+'","auto": 1}'))

editor.placeBlock((base_x,-63,base_z), Block('minecraft:command_block',
        data='{"Command": "clear @p","auto": 1})'))

start_tag='summon_monster'
end_tag='summon_monster_done'
z = enemy_pos[0]
x = enemy_pos[1]

base_condi='tag=!'+start_tag+',tag=!'+end_tag
for tag in tag_list:
    base_condi+=',tag='+tag

command1 = ('execute as @a['+base_condi+'] run tag @s add '+start_tag)
command2 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] run tellraw @s '
            '{\\\"text\\\":\\\"Defeat the enemy\\\",\\\"color\\\":\\\"green\\\"}')
command3 = (f'execute as @a[tag={start_tag},tag=!{end_tag}] run summon '
            f'minecraft:zombie {base_x + x} {base_y+2} {base_z + z}')
command4 = ('execute as @a[tag='+start_tag+',tag=!'+end_tag+'] if entity @e[type=minecraft:zombie, distance=..5] run tag @s add '+end_tag)
editor.placeBlock((base_x + x, base_y, base_z + z),
                    Block('minecraft:repeating_command_block', states={'facing': 'down'},
        data=f'{{"Command": "{command1}","auto": 1}}'))
editor.placeBlock((base_x + x, base_y-1, base_z + z),
                    Block('minecraft:chain_command_block', states={'facing': 'down'},
        data=f'{{"Command": "{command2}","auto": 1}}'))
editor.placeBlock((base_x + x, base_y-2, base_z + z),
                    Block('minecraft:chain_command_block', states={'facing': 'down'},
        data=f'{{"Command": "{command3}","auto": 1}}'))
editor.placeBlock((base_x + x, base_y-3, base_z + z),
                    Block('minecraft:chain_command_block', states={'facing': 'down'},
        data=f'{{"Command": "{command4}","auto": 1}}'))