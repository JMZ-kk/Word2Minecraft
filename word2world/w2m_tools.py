import openai
from configs import Config
from utils import map_to_list, batch_similarity, extract_list, extract_dict, extract_between_ticks
from lookup import BLOCKS, FILTERING
from small_buildings import *
from Objective import Objective
from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
import random
import openai
import base64

cfg = Config()

def extract_tile(tile, choices):
    cfg = Config()
    t_prompt = (f'Given a tile A: \n{tile}\n, '
                f'and a list B: {choices},'
                f'Please find the element in list B that has the closest meaning with A. '
                f'Return only the item in list B, no other information is needed')
    t_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": t_prompt}
    ], temperature=1)
    print('name: ' + t_discriptions)
    return t_discriptions


def extract_tile_set(char_tile_mapping, choices, item_check=False):
    cfg = Config()
    tile_list = list(char_tile_mapping.keys())
    tileset_prompt = (f'Given a list A: \n{tile_list}\n, '
                      f'and a list B: {choices},'
                      f'for each element in list A, please find the element in list B that has the closest meaning. '
                      f'Return only a Python Dictionary where each key is an element from list A, and '
                      f'its value is the most semantically similar element from list B. All value should be different')
    print(tileset_prompt)
    for i in range(3):
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

def extract_goal_order(goal_des,objective_list):
    cfg = Config()
    order_prompt = (f'You will be given a list of objectives: \n{objective_list}\n '
                   f'and a description of them: \n{goal_des}\n. You need to determine the order of these'
                   f'tasks. Please return a python dictionary where the key is objective '
                   f'and the value is an integer (start from 0) representing the corresponding order.'
                   f'Two objectives are allowed to share the same integer.')
    order_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": order_prompt}
    ], temperature=1)
    order_dict = extract_dict(order_discriptions['choices'][0]['message']['content'])
    return order_prompt, order_discriptions, order_dict

def repair_goals(goal_prompt, goal_discriptions, old_list, cate_list,nf_list=None):
    cfg = Config()
    repair_prompt = (
        f'For those objectives \n{old_list}\n do not fit into any of these categories, could you slightly change the'
        f'objectives to fit them into one of the category in \n{cate_list}\n?'
        f'Please return a python dict where the key is the old objective and '
        f'the value is a python list: [repaired objective,category].')
    if nf_list is not None and len(nf_list)>0:
        repair_prompt+=f'Try to create more objectives in {nf_list}'
    repair_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": goal_prompt},
        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": repair_prompt}
    ], temperature=1)
    repair_dict = extract_dict(repair_discriptions['choices'][0]['message']['content'])
    return repair_prompt, repair_discriptions, repair_dict


def extract_npc(goal_prompt, goal_discriptions, objective, tool_list):
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


def pick_scaling_tiles(des2not, tile_map, story, story_prompt):
    cfg = Config()
    pick_building_prompt = (f'Given the story, a 2D map \n{tile_map}\n, and a dictionary {des2not} where each key is a tile\'s '
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
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
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
                       f'Please provide the output as a Python list of tiles.'
                       f'Please make sure that each tile maps to one of the list.')
    sub_tile_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": sub_tile_prompt}
    ], temperature=1)
    sub_tile_list = extract_list(sub_tile_discriptions['choices'][0]['message']['content'])
    import string
    sub_tile_dict = dict(zip(string.ascii_lowercase, sub_tile_list))
    return sub_tile_dict, sub_tile_discriptions, sub_tile_prompt


def subworld_generation(story, story_prompt, sub_tile_discriptions, sub_tile_prompt, sub_tile_dict):
    sub_world_prompt = (f'You are given a tile dictionary \n{sub_tile_dict}\n, the key is notation and'
                        f'the value is the tile. Please create a map for the objective by notations, '
                        f'which should contain at least 20 rows and 20 columns. Please return a python string list, where'
                        f'each element represents a line. Please make sure that all characters used are notations in the tile dictionary. ')
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
    return padded_lines, sub_world_discriptions, sub_world_prompt

def sub_rule_gen(story, story_prompt, sub_world,sub_world_discriptions, sub_world_prompt, sub_tile_discriptions, sub_tile_prompt):
    sub_rule_prompt = (f'You are given a map \n{sub_world}\n. Please create a rule for it based on its objective.'
                        f'The rule should include the condition of fulfilling this objective. '
                       f'Please only return the rule.')
    sub_rule_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": sub_tile_prompt},
        {"role": "assistant", "content": sub_tile_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_world_prompt},
        {"role": "assistant", "content": sub_world_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_rule_prompt}
    ], temperature=1)
    return sub_rule_prompt, sub_rule_discriptions

def sub_cdblocks(story, story_prompt, sub_rule_prompt, sub_rule_discriptions,sub_world,sub_world_discriptions, sub_world_prompt, sub_tile_discriptions, sub_tile_prompt):
    sub_cdblocks_prompt = (f'The map \n{sub_world}\n will be implmented in Minecraft from (0,1,0) to ({len(sub_world)},1,{len(sub_world[0])}.)'
                       f'You can put any numbers of repeating_command_block, chain_command_block, command_block from (0,0,0) to ({len(sub_world)},-9,{len(sub_world[0])}.).'
                       f'Please return a python dictionary, whose key is the (x,y,z,type), in which (x,y,z) is coordinate and type is one of '
                       f'repeating_command_block, chain_command_block, command_block), and value is the command'
                       )
    sub_cdblocks_discriptions = openai.ChatCompletion.create(model=cfg.model, messages=[
        {"role": "user", "content": story_prompt},
        {"role": "assistant", "content": story},
        {"role": "user", "content": sub_tile_prompt},
        {"role": "assistant", "content": sub_tile_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_world_prompt},
        {"role": "assistant", "content": sub_world_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_rule_prompt},
        {"role": "assistant", "content": sub_rule_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": sub_cdblocks_prompt}
    ], temperature=1)
    ans=extract_dict(sub_cdblocks_discriptions['choices'][0]['message']['content'])
    return ans

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


def update_map(check, biggest_m, biggest_n, block_size, tar, grid_world,check4show=None):
    for i in range(block_size):
        for j in range(block_size):
            check[biggest_m + i][biggest_n + j] = 5
            if check4show is not None:
                check4show[biggest_m + i][biggest_n + j] = 4
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


subtile_replace = {}


def place_block(editor, x, y, z, tile, block_data):
    if tile not in block_data and tile not in subtile_replace:
        print('old: ' + tile)
        similarities = batch_similarity([tile] * len(block_data),
                                        block_data, model_type='bert')
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        new_tile = block_data[max_index]
        print('new: ' + new_tile)
        subtile_replace[tile] = new_tile
    if tile in subtile_replace:
        tile = subtile_replace[tile]
    block = 'minecraft:' + tile
    if 'leave' in block:
        editor.placeBlock((x, y - 1, z), Block('minecraft:oak_wood'))
    elif 'lily_pad' in block:
        editor.placeBlock((x, y - 1, z), Block('minecraft:water'))
        editor.placeBlock((x, y - 2, z), Block('minecraft:dirt'))
    else:
        editor.placeBlock((x, y - 1, z), Block('minecraft:dirt'))
    if block in FILTERING:
        editor.placeBlock((x, y - 1, z), Block(block))
        editor.placeBlock((x, y - 2, z), Block('minecraft:dirt'))
    elif 'cactus' in block:
        editor.placeBlock((x, y - 1, z), Block('minecraft:dirt'))
        editor.placeBlock((x, y, z), Block('minecraft:sand'))
        editor.placeBlock((x, y + 1, z), Block(block))
    else:
        editor.placeBlock((x, y, z), Block(block))


def countdown(editor, timer, tag, prompt_word, x, start_y, z):
    time_command_2 = 'execute if score global Tick matches 19 run scoreboard players remove @a[scores={' + timer + '=1..}] ' + timer + ' 1'
    time_command_3 = 'title @a[scores={' + timer + '=1..},tag=' + tag + '] actionbar {\\\"text\\\":\\\"Countdown: \\\",\\\"color\\\":\\\"yellow\\\",\\\"extra\\\":[{\\\"score\\\":{\\\"name\\\":\\\"@a\\\",\\\"objective\\\":\\\"' + timer + '\\\"}}]}'
    time_command_4 = 'execute as @a[scores={' + timer + '=0},tag=' + tag + '] run title @a title {\\\"text\\\":\\\"' + prompt_word + '\\\",\\\"color\\\":\\\"red\\\"}'
    editor.placeBlock((x, start_y, z),
                      Block('minecraft:repeating_command_block', states={'facing': 'down'},
                            data=f'{{"Command": "{time_command_2}","auto": 1}}'))
    editor.placeBlock((x, start_y - 1, z),
                      Block('minecraft:chain_command_block', states={'facing': 'down'},
                            data=f'{{"Command": "{time_command_3}","auto": 1}}'))
    editor.placeBlock((x, start_y - 2, z),
                      Block('minecraft:chain_command_block', states={'facing': 'down'},
                            data=f'{{"Command": "{time_command_4}","auto": 1}}'))
from PIL import Image
import base64
import io

def encode_image(image_path, max_size=(256, 256)):
    image = Image.open(image_path)
    image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded_str

def get_common_advice(image_path, prompt):
    cfg=Config()

    base64_image = encode_image(image_path)

    response = openai.ChatCompletion.create(
        model=cfg.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ], temperature=1
    )

    return response.choices[0].message.content

