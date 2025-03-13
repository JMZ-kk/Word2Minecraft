import os
import json
import numpy as np
import matplotlib.pyplot as plt
from w2m_tools import *

model_list=['mini','example']
x_cor=['']
dir='word2world/examples/dataset'
res={}
eval_system_prompt = (f"You are an evaluator of a 2D tilemap world created from a story. "
                      f"You are provided by a story and a "
                      f"Python dictionary-like mapping of tiles and characters or alphabets. "
                      f"You evaluate based on how the tiles have been placed and if they "
                      f"match how the story has explained. ")
for model_name in model_list:
    values=[]
    num=0
    for filename in os.listdir(dir):
        if filename.startswith(model_name) :
            with open(f'{dir}/{filename}', 'r') as file:
                data = json.load(file)
                world=data['round_0']['world']
                story=data['round_0']['story']
                tile_map_dictionary=data['round_0']['tile_mapping']
                evaluation_prompt = (f"Given the story, tiles used to create the 2D map, "
                                     f"and the 2D map, suggest whether the 2D tilemap "
                                     f"world is coherent to the story. The story is as "
                                     f"follows:\n{story}\nThe tile mapping:{tile_map_dictionary}\n"
                                     f"The 2D tilemap world:\n{world}\n Check for all "
                                     f"the tiles mentioned in the tile mapping being in "
                                     f"the 2D tilemap world.")
                evaluation_prompt_2 = (f"Return a coherence score from 1 to 100, "
                                       f"where 1 means the story and map are "
                                       f"completely different, and 100 means "
                                       f"they are nearly identical. Please only return an integer, "
                                       f"none additional information shall be added.")
                world_eval = openai.ChatCompletion.create(model="gpt-4-turbo-2024-04-09", messages=[
                    {"role": "system", "content": eval_system_prompt},
                    {"role": "user", "content": evaluation_prompt}
                ],temperature=0.6)
                while True:
                    try:
                        world_eval_2 = openai.ChatCompletion.create(model="gpt-4-turbo-2024-04-09", messages=[
                            {"role": "system", "content": eval_system_prompt},
                            {"role": "user", "content": evaluation_prompt},
                            {"role": "assistant", "content": world_eval['choices'][0]['message']['content']},
                            {"role": "user", "content": evaluation_prompt_2}
                        ], temperature=0.6)
                        # print(world_eval['choices'][0]['message']['content'])
                        print(world_eval_2['choices'][0]['message']['content'])
                        score=int(world_eval_2['choices'][0]['message']['content'])
                        values.append(score)
                        break
                    except:
                        print('bug')
                print()
    res[model_name] = {'mean': np.mean(values), 'std': np.std(values, ddof=0)}

models = list(res.keys())
means = [res[m]['mean'] for m in models]
std_devs = [res[m]['std'] for m in models]

plt.rcParams.update({'font.size': 15})
x = np.arange(len(models))
plt.bar(x, means, yerr=std_devs, capsize=5, color=['r','orange'], alpha=0.9, hatch='/')
plt.xticks(x, models, rotation=15)
plt.ylabel("Value")
plt.title("Coherence Score")

plt.show()