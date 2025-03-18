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

# collect message of original map
df = pd.read_csv('word2world/items.csv', header=None)
df2 = pd.read_csv('word2world/enemy.csv', header=None)
df3 = pd.read_csv('word2world/tools.csv', header=None)
item_list = df[0].tolist()
enemy_list = df2[0].tolist()
tool_list = df3[0].tolist()

dir='word2world/examples/dataset'
model_name='t86'
for filename in os.listdir(dir):
    if filename.startswith(model_name):
        try:
            with open(f'{dir}/{filename}', 'r') as file:
                data = json.load(file)
            round_number = "round_1"
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

            last_time=0
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
            obj_des_list=[]
            for obj in objective_list:
                obj_des_list.append(obj.description)
                try:
                    if goal_dict[obj.description] == 'Other':
                        false_des.append(obj.description)
                    else:
                        obj.goal = goal_dict[obj.description]
                except:
                    pass
            if len(false_des) > 0:
                repair_prompt, repair_discriptions, repair_dict = repair_goals(goal_prompt, goal_discriptions, false_des,
                                                                               objective_types)
                for obj in objective_list:
                    for key in repair_dict.keys():
                        if obj.description == key:
                            obj.description = repair_dict[key][0]
                            obj.goal = repair_dict[key][1]

            clist=[]
            for obj in objective_list:
                if obj.goal == 'Navigate the maze to find an exit':
                    obj.map = generate_maze_array()
                    tile_list = extract_submaze_tile(story, story_prompt, obj.description, block_data)
                    obj.tile_dict={}
                    obj.tile_dict['0']=tile_list[0]
                    obj.tile_dict['1'] = tile_list[1]
                elif obj.goal=='Chat with NPC':
                    obj.npc_info, npc_prompt, npc_discriptions=extract_npc(goal_prompt, goal_discriptions, obj.description,tool_list)
                elif obj.goal=='Defeat the enemy':
                    enemy = [k for k, v in data[round_number]["tile_mapping"].items() if v == '#'][0]
                    obj.enemy_name=extract_enemy(enemy, enemy_list)
                else:
                    obj.tile_dict, sub_tile_discriptions, sub_tile_prompt = extract_subworld_tile(story, story_prompt,
                                                                                                  obj.description, block_data)
                    obj.map,sub_world_discriptions,sub_world_prompt=subworld_generation(story, story_prompt,sub_tile_discriptions, sub_tile_prompt,obj.tile_dict)
                    if obj.goal=='Collect Items':
                        clist.append(obj.description)
            if len(clist) > 0:
                item_dict = extract_item(goal_prompt, goal_discriptions, clist,block_data)
                for key in item_dict.keys():
                    for obj in objective_list:
                        if obj.description == key:
                            obj.item = item_dict[key]

            occupied_positions = []
            objectives_2_map(grid_world, objective_list, walkable_tile_counts, occupied_positions)

            new_obj={}
            for obj in objective_list:
                new_obj[obj.description]=obj.coordinates

            check = np.zeros((len(grid_world), len(grid_world[0])))
            for i in range(len(grid_world)):
                for j in range(len(grid_world[0])):
                    if (i, j) in occupied_positions:
                        check[i][j] = 3
                    elif f'''{grid_world[i][j]}''' not in walkables:
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


            tileset, tileset_discriptions, tileset_prompt = extract_tile_set(non_special_char_map, block_data)

            #------------------------------------------scaling problem--------------------------------------
            scaling_tiles_list, pick_scaling_discriptions, pick_scaling_prompt = pick_scaling_tiles(bottom_half_des2not, grid_world,story, story_prompt)
            size_dict, size_discriptions, extract_size_prompt = extract_scaling_size(scaling_tiles_list, pick_scaling_discriptions,
                                                                                     pick_scaling_prompt)

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
            new_file={}
            new_file['world']=grid_world
            swapped_dict = {v: k for k, v in tileset.items()}
            new_file['tile_mapping']=swapped_dict
            new_file['objectives']=objectives
            new_file['old_tile_mapping']=char_tile_mapping
            new_file['compo_dict']=compo_dict
            new_file['new_obj']=new_obj
            # with open(cfg.save_dir + f"/new_{filename}.json", 'w') as f:
            #     json.dump(new_file, f)

            editor = Editor(buffering=False)
            base_map = grid_world

            chasing_mode = False
            summon_block_num = 0
            sum_x_len = len(grid_world)
            finished_scaling = np.zeros((len(grid_world), len(grid_world[0])))

            placement={}
            res=[]
            for x in range(len(base_map)):
                col=[]
                for z in range(len(base_map[x])):
                    if (x, z) in occupied_positions:
                        for obj in objective_list:
                            if obj.coordinates == (x, z):
                                col.append([obj.description])
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
                                col.append(tuple[3])
                        if z == player_pos[0] and x == player_pos[1]:
                            col.append([[k for k, v in data[round_number]["tile_mapping"].items() if v == '@'][0]])
                        elif z == enemy_pos[0] and x == enemy_pos[1]:
                            col.append([[k for k, v in data[round_number]["tile_mapping"].items() if v == '#'][0]])
                        else:
                            try:
                                tile = tileset.get(base_map[x][z])
                                block = 'minecraft:' + tile
                                if base_map[x][z] not in walkables and block not in FILTERING:
                                    col.append([tile + '_barrier'])
                                else:
                                    col.append([tile])
                            except:
                                col.append([default_walkable_tile])
                res.append(col)
            blockfile_name=f'block_{filename}'
            with open('word2world/examples/blocks/'+blockfile_name, "w") as f:
                json.dump(res, f, indent=2)
                print(f"Blocks saved to {'word2world/examples/blocks/'+blockfile_name}")
        except:
            print('bug')
