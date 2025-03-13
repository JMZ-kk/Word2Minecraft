import os
import json
import numpy as np
import matplotlib.pyplot as plt
from w2m_tools import *

model_list=['mini','example']
x_cor=['']
dir='word2world/examples/dataset'
img_dir='word2world/images'
res={}
eval_system_prompt = (f"You are an evaluator of a 2D tilemap world created from a story. "
                      f"You are provided by a game level."
                      f"You evaluate based on how the tiles have been placed and if they "
                      f"match how the story has explained. ")

def img_ana(filename):
    with open(f'{dir}/{filename}.json', 'r') as file:
        data = json.load(file)
        world = data['round_0']['world']
        story = data['round_0']['story']

        cfg = Config()

        base64_image = encode_image(f'{img_dir}/{filename}_tpd.png')
        evaluation_prompt=(f"Given the story, tiles used to create the 2D map, "
                            f"and the 2D map, suggest whether the 2D tilemap "
                            f"world is coherent to the story. The story is as "
                            f"follows:\n{story}\nThe top down view:\n{base64_image}\n")
        evaluation_prompt_2 = (f"Return a coherence score from 1 to 100, "
                               f"where 1 means the story and map are "
                               f"completely different, and 100 means "
                               f"they are nearly identical. Please only return an integer, "
                               f"none additional information shall be added.")
        world_eval = openai.ChatCompletion.create(
            model=cfg.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Given the story, tiles used to create the 2D map, "
                                     f"and the 2D map, suggest whether the 2D tilemap "
                                     f"world is coherent to the story. The story is as "
                                     f"follows:\n{story}\nThe top down view:\n{base64_image}\n"
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
                score = int(world_eval_2['choices'][0]['message']['content'])
                return score
            except:
                print('bug')


filename='example_1'
img_ana(filename)

# for model_name in model_list:
#     values=[]
#     num=0
#     for filename in os.listdir(dir):
#         if filename.startswith(model_name) :
#             img_ana(filename)
#     res[model_name] = {'mean': np.mean(values), 'std': np.std(values, ddof=0)}

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