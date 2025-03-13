from story_tools import *
import os
import json

model_name='mini'
block_dir='word2world/examples/blocks'
dir='word2world/examples/dataset'
for filename in os.listdir(dir):
    try:
        if filename.startswith(model_name):
            block_filename='block_'+filename
            with open(dir+'/'+filename,'r') as f:
                data_0=json.load(f)
                objs=json.dumps(data_0['round_1']['objectives'])

            with open(f'{block_dir}/{block_filename}', 'r') as file_0:
                data = json.load(file_0)
                width=len(data)
                length=len(data[0])
                json_str = json.dumps(data)
            story_gen_discriptions, story_gen_prompt=story_generator_3D(json_str,2,width,length,objs, model=cfg.model)
            tt=filename.replace('json','txt')
            with open(f'word2world/story/story_{tt}', 'w', encoding='utf-8') as file:
                file.write(story_gen_discriptions['choices'][0]['message']['content'])
            print(story_gen_discriptions['choices'][0]['message']['content'])
    except:
        pass