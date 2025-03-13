import os
import json
import numpy as np
import matplotlib.pyplot as plt

model_list=['gpt4o','mini','turbo']
dir='word2world/examples/dataset'
res={}
for model in model_list:
    tset = {}
    for filename in os.listdir(dir):
        if filename.startswith(model):
            with open(f'{dir}/{filename}', 'r') as file:
                data = json.load(file)
                evaluation=data['round_1']['evaluations']
                for key in evaluation.keys():
                    if key not in tset:
                        tset[key]=[evaluation[key]]
                    else:
                        tset[key].append(evaluation[key])
    res[model] = {key: {'mean': np.mean(values), 'std': np.std(values, ddof=0)} for key, values in tset.items()}

models = list(res.keys())
means = [res[m]['Accuracy_of_important_tiles_placed']['mean'] for m in models]
std_devs = [res[m]['Accuracy_of_important_tiles_placed']['std'] for m in models]

plt.rcParams.update({'font.size': 16})
x = np.arange(len(models))
plt.bar(x, means, yerr=std_devs, capsize=5, color='navy', alpha=0.9, hatch='/')
plt.xticks(x, models, rotation=15)
plt.ylabel("Value")
plt.title("Accuracy of important tiles")

plt.show()