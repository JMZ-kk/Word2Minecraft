import numpy as np
from collections import Counter
import os
import json
import matplotlib.pyplot as plt

def compute_tile_variety(map_array):
    unique_tiles, counts = np.unique(map_array, return_counts=True)
    tile_count = len(unique_tiles)
    total_tiles = map_array.size
    probabilities = counts / total_tiles
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return tile_count, entropy


def parse_map_string(map_string):
    map_lines = map_string.strip().split("\n")
    map_array = np.array([list(line) for line in map_lines])
    return map_array


dir='word2world/examples/dataset'
dir_2='word2world/examples'
suffix='json'
count_list=[]
entropy_list=[]
for filename in os.listdir(dir):
    # if filename.endswith(suffix):
    if 'mini' in filename:
        with open(f'{dir}/{filename}', 'r') as file:
            data = json.load(file)
            round_num=''
            if 'round_1' in data:
                round_num = 'round_1'
            elif 'round_0' in data:
                round_num = 'round_0'
            else:
                continue
            world_1st_layer = data[round_num]["world_1st_layer"]["world"]
            lines = world_1st_layer.strip().split("\n")
            max_length = max(len(line) for line in lines)
            padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in lines]
            map_array = np.array(padded_lines)
            tile_count, entropy = compute_tile_variety(map_array)
            print(f"Tile种类数: {tile_count}")
            print(f"Shannon Entropy: {entropy:.4f}")
            count_list.append(tile_count)
            entropy_list.append(entropy)

count_mean, count_std = np.mean(count_list), np.std(count_list)
entropy_mean, entropy_std = np.mean(entropy_list), np.std(entropy_list)

count_list=[]
entropy_list=[]
for filename in os.listdir(dir_2):
    if filename.endswith(suffix):
        with open(f'{dir_2}/{filename}', 'r') as file:
            data = json.load(file)
            round_num=''
            if 'round_1' in data:
                round_num = 'round_1'
            elif 'round_0' in data:
                round_num = 'round_0'
            else:
                continue
            world_1st_layer = data[round_num]["world_1st_layer"]["world"]
            lines = world_1st_layer.strip().split("\n")
            max_length = max(len(line) for line in lines)
            padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in lines]
            map_array = np.array(padded_lines)
            tile_count, entropy = compute_tile_variety(map_array)
            print(f"Tile种类数: {tile_count}")
            print(f"Shannon Entropy: {entropy:.4f}")
            count_list.append(tile_count)
            entropy_list.append(entropy)
print(count_list)
print(entropy_list)
count_mean_turbo, count_std_turbo = np.mean(count_list), np.std(count_list)
entropy_mean_turbo, entropy_std_turbo = np.mean(entropy_list), np.std(entropy_list)

plt.rcParams.update({'font.size': 15})

# 设置子图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 定义柱状图的宽度
bar_width = 0.4
opacity = 0.8
x = np.arange(1)  # x 轴索引（单个柱子组）

# **子图 1：Tile Count**
axes[0].bar(x - bar_width/2, count_mean, bar_width, yerr=count_std,
            capsize=5, label="gpt-4o-mini", color='r', alpha=opacity)
axes[0].bar(x + bar_width/2, count_mean_turbo, bar_width, yerr=count_std_turbo,
            capsize=5, label="gpt-4-turbo", color='orange', alpha=opacity)
axes[0].set_xticks([])
axes[0].set_title("Tile Count")
axes[0].set_ylabel("Count Average")
axes[0].legend()
axes[0].grid(axis='y', linestyle="--", alpha=0.6)

# **子图 2：Shannon Entropy**
axes[1].bar(x - bar_width/2, entropy_mean, bar_width, yerr=entropy_std,
            capsize=5, label="gpt-4o-mini", color='r', alpha=opacity)
axes[1].bar(x + bar_width/2, entropy_mean_turbo, bar_width, yerr=entropy_std_turbo,
            capsize=5, label="gpt-4-turbo", color='orange', alpha=opacity)
axes[1].set_xticks([])
axes[1].set_title("Shannon Entropy")
axes[1].set_ylabel("Entropy Average")
axes[1].legend()
axes[1].grid(axis='y', linestyle="--", alpha=0.6)

# 调整布局
plt.tight_layout()
plt.show()





#
# plt.figure(figsize=(12, 5))
#

#
# # Tile 种类数直方图
# plt.subplot(1, 2, 1)
# plt.hist(count_list, bins=15, alpha=0.7, color='b', edgecolor='black')
# plt.xlabel('Tile type count')
# plt.ylabel('Frequency')
# plt.title('Tile count distribution')
#
# # 熵值直方图
# plt.subplot(1, 2, 2)
# plt.hist(entropy_list, bins=15, alpha=0.7, color='g', edgecolor='black')
# plt.xlabel('Shannon Entropy')
# plt.ylabel('Frequency')
# plt.title('Shannon Entropy distribution')
#
# plt.tight_layout()
# plt.show()