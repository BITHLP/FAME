import pandas as pd
import json
import random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader



src_file_name_ls = ["/data/lzeng/editllm_dataset/FAME/code/model/result_data/QueryDataset6.json",]
meta_file_name = "/data/lzeng/editllm_dataset/result_1/script/source2/metadata.json"

print(src_file_name_ls)
def delete_lines(text, num_lines):
    lines = text.split('\n')    
    del lines[:num_lines]    
    result = '\n'.join(lines)    
    return result
model_stats = {}
num_samples = 10  

import string
meaningless_words = ["the", "a", "an", "in", "on", "at", "of", "to", "with", "by", "for", "as", "this",
                     "that", "is", "are", "was", "were", "am", "be", "being", "been", "it", "this",
                     "his", "its", "who", "what", "when", "where", "why", "how", "which", ""]

def remove_prefix(prefix, main_string):
    if main_string.startswith(prefix):
        main_string = main_string[len(prefix):]
    return main_string


def normal_output(output):
    output = output.strip().split('\n')[0]
    
    
    output = remove_prefix('A:',output)
    output = remove_prefix('Answer:',output)
    output = remove_prefix(':',output)
    return output

def evaluate_output(corrected_fact, output):

    corrected_fact = corrected_fact.strip().lower()
    output = output.strip().lower()
    output = output.split('\n')[0]

    translator = str.maketrans('', '', string.punctuation + '\\')
    corrected_fact = corrected_fact.translate(translator)
    output = output.translate(translator)
    output = output.strip().lower()



    if corrected_fact == output[:len(corrected_fact)]:
        return "completely_identical"
    elif corrected_fact in output:
        return "contains_correct_answer"
        if len(output.split()) <= 3:
            return "contains_correct_answer_short"
        else:
            return "contains_correct_answer"
    elif all(char in meaningless_words for char in output):
        return "meaningless_output"
    else:
        return "answer_incorrect"
    

output_categories_seen = {
    "completely_identical": False,
    "contains_correct_answer": False,
    "contains_correct_answer_short": False,
    "output_excess_short": False,
    "meaningless_output": False,
    "answer_incorrect": False,
}

if __name__ == "__main__":

    
    with open(src_file_name_ls[0], 'r') as file:
        data = json.load(file)
    
    for i in range(1,len(src_file_name_ls)):
        with open(src_file_name_ls[i], 'r') as app_file:
            app_data = json.load(app_file)
        data = [
            {
                **data[index],
                "model_outputs":{
                    **data[index]["model_outputs"],
                    **app_data[index]["model_outputs"]
                }
            }

            for index in range(len(data))
        ]

        
    with open(meta_file_name, 'r') as file1:
        meta = json.load(file1)

    ans_type_ls = [list(i.keys()) for i in list(data[0]['model_outputs'].values())][0]
    print(ans_type_ls)
    for ans_type in ans_type_ls:
        for _index,item in enumerate(data):
            try:
                model_outputs = item['model_outputs']
            except KeyError:
                continue
            corrected_fact = item['objectLabel']
            if ans_type == "local":
                corrected_fact = item['localityobjectLabel']

            data[_index]["processed_output"] = {
                model:{
                }
                for model in model_outputs.keys()
            }
            data[_index]["model_result"] = {
                model:{
                }
                for model in model_outputs.keys()
            }

            for model, output in model_outputs.items():
                if isinstance(output,dict):
                    output = output[ans_type]
                if isinstance(output,dict):
                    inp = output['inp']
                    output = output['content']
                    output = remove_prefix(inp,output)
                if isinstance(output,list):
                    output = output[0]

                output = normal_output(output)
                

                data[_index]['processed_output'][model][ans_type] = output
                output_category = evaluate_output(corrected_fact, output)

                if model in model_stats:
                    model_stats[model][output_category] += 1
                else:
                    model_stats[model] = {
                        "completely_identical": 0,
                        "contains_correct_answer_short": 0,
                        "contains_correct_answer": 0,
                        "output_excess_short": 0,
                        "meaningless_output": 0,
                        "answer_incorrect": 0,
                    }
                    model_stats[model][output_category] += 1
            
                data[_index]["model_result"][model][ans_type] = output_category


        print(ans_type)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame.from_dict(model_stats, orient='index')

        print(df)
        print("*"*50)
        model_stats = {}


        with open(src_file_name_ls[0][:-5] + "_ed.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
