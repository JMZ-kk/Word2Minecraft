from story_tools import *
import os
import json
from fixers import pad_rows_to_max_length

model_name='example'
block_dir='word2world/examples/blocks'
dir='word2world/examples/dataset'

for filename in os.listdir(dir):
    try:
        if filename.startswith(model_name) :
            with open(f'{dir}/{filename}', 'r') as file:
                data = json.load(file)
                grid_str = data['round_0']['world']
                grid_str = pad_rows_to_max_length(grid_str)
                grid_world_0 = map_to_list(grid_str)  # old map
                tile_map=data['round_0']["tile_mapping"]
                story_gen_discriptions, story_gen_prompt=story_generator(grid_world_0,tile_map, model=cfg.model)
                tt = filename.replace('json', 'txt')
                tt = tt.replace('example', 'turbo')
                with open(f'word2world/story/rc_{tt}', 'w', encoding='utf-8') as file:
                    file.write(story_gen_discriptions['choices'][0]['message']['content'])
    except Exception as e:
        print('bug')