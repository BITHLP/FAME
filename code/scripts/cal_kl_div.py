import os
import torch
from torch.nn.functional import kl_div
import numpy as np
import json
import random 
from tqdm import tqdm
from datetime import datetime
import os
def read_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, fn):
    if data is None:
        print("data is None")
        return
    else:
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=True, indent=4)


def calculate_kl_divergence(folder_path_a, folder_path_b):
    files_a = os.listdir(folder_path_a)
    files_b = os.listdir(folder_path_b)
    files_a = files_a[:1530]
    assert len(files_a) == len(files_b), f"Number of files in both folders must be equal, get {len(files_a)} and {len(files_b)}"

    kl_divergences = []
    folder_info = {
        'folder_path_a': folder_path_a,
        'folder_path_b': folder_path_b,
        'num_files': len(files_a),
    }

    for i in tqdm(range(len(files_a))):
        file_a, file_b = files_a[i], files_b[i]
        file_path_a = os.path.join(folder_path_a, file_a)
        file_path_b = os.path.join(folder_path_b, file_b)

        if file_a.endswith('.pt') and file_b.endswith('.pt'):
            a = torch.load(file_path_a,map_location = "cpu")
            b = torch.load(file_path_b,map_location = "cpu")


            kl_value = kl_div(
                -a, -b, log_target=True,
                reduction="batchmean"
            ).cpu().numpy()
            kl_divergences.append(kl_value)
    average_kl_divergence = np.mean(kl_divergences)

    kl_divergences = [
        i.tolist() for i in kl_divergences
    ]
    result_dict = {
        **folder_info,
        'average_kl_divergence': average_kl_divergence.tolist(),
        'kl_divergences': kl_divergences,
    }
    print(average_kl_divergence)

    return result_dict
folder_path_a = '/data/lzeng/editllm_dataset/result_final_1212/edit_model/result_data/QueryDataset97.json'
folder_path_b = '/data/lzeng/Easyedit_org/EasyEdit/result_data/QueryDataset129.json'

folder_path_a = folder_path_a.replace(".json","")+"/prob_dir"
folder_path_b = folder_path_b.replace(".json","")+"/prob_dir"

print(folder_path_a)
print(folder_path_b)

result = calculate_kl_divergence(folder_path_a, folder_path_b)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
kl_div_base = "/data/lzeng/editllm_dataset/result_final_1212/script/kl_div"
output_file_path = f"{kl_div_base}/{current_time}.json"
write_json(result,output_file_path)
