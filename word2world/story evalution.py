import numpy as np

from story_tools import *
import os
import json

import matplotlib.pyplot as plt

model_list=['mini','turbo']
story_dir='word2world/story'
dir='word2world/examples/dataset'
res={}

for model_name in model_list:
    res[model_name] =[]
    for filename in os.listdir(dir):
        try:
            if model_name=='turbo':
                if 'example' in filename:
                    story_filename = 'rc_' + (filename.replace('json', 'txt'))
                    story_filename = 'rc_' +(story_filename.replace('example', 'turbo'))
                    similarity = story_comparator_embd(story_1, story_2)
                    res[model_name].append(similarity)
                    print('turbo')
            elif model_name in filename:
                story_filename = 'rc_' + (filename.replace('json', 'txt'))
                with open(dir+'/'+filename,'r') as f:
                    data_0=json.load(f)
                    story_1=json.dumps(data_0['round_0']['story'])

                with open(story_dir+'/'+story_filename, "r", encoding="utf-8") as f:
                    story_2 = f.read()

                similarity=story_comparator_embd(story_1,story_2)
                res[model_name].append(similarity)
                print('mini')
        except Exception as e:
            pass

plt.rcParams.update({'font.size': 15})
x = np.arange(len(model_list))
means=[]
stds=[]
for model_name in model_list:
    means.append(np.mean(res[model_name]))
    stds.append(np.std(res[model_name]))
plt.bar(x, means, yerr=stds, capsize=5, color=['r','orange'], alpha=0.9)
plt.xticks(x, model_list, rotation=15)
plt.ylabel("Similarity Score")
plt.title("Story coherence")

plt.show()
