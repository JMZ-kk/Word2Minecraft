import json
import glob
import os
import numpy as np


def calculate_one_ratio(data):
    total_elements = sum(len(row) for row in data)
    count_ones = sum(row.count(1) for row in data)
    return count_ones / total_elements if total_elements > 0 else 0


def load_json_files():
    file_paths = glob.glob("data*.json")
    ratios = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            ratios.append(calculate_one_ratio(data))

    return ratios


ratios = load_json_files()
if ratios:
    mean_ratio = np.mean(ratios)
    var_ratio = np.std(ratios)
    print(f"1 的占比平均值: {mean_ratio:.4f}")
    print(f"1 的占比std: {var_ratio:.4f}")
else:
    print("未找到匹配的 JSON 文件。")