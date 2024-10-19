import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForCausalLM, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast
)
import typing
import json
from tqdm import tqdm
from datetime import datetime
import os
import gc
import random
import time
import skeme
import argparse

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_name': ['edit_llama-2-7b-hf'],
    'output_file': "",
    'data_dir': "./dsets",
    'batch_size': 24,
    'few_shot_count': 3,
    'log_filename': None,
    'types': ["choose"],
    # 'types': ["completion", "qa", "fill", "choose","fc"],
    "model_type": "edited",
    "sys_prompt": {
        "completion": "Completion the sentence with a prase.",
        "qa": "Answer the question with one prase.",
        "local": "Answer the question with one prase.",
        "fill": "Identify the content within the parentheses and provide the missing information.",
        "choose": "Choose the best answer.",
        "dialog": "Answer the question with one prase",
        "triviaQA":"",
        "fc": "Determine the veracity of the provided statement. Clearly output 'True' if the statement is accurate and 'False' if it is not."
    },
    "alpaca_prompt": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{sys_prompt}
""",
    "alpaca_prompt_few_shot": """
### Input:
{query}

### Response:{ans}\n""",

}



def create_log_file():
    current_time = datetime.now().strftime("%m-%d-%H-%M")
    log_filename = f"./log/log_{current_time}.txt"

    try:
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Log created at {current_time}\n")
        print(f"Log file '{log_filename}' created successfully.")
        config['log_filename'] = log_filename
    except Exception as e:
        print(f"Error creating log file: {str(e)}")


def treat_log_info(log_ls, log_dict):

    try:
        ppl = [subelement for
               sublist in [i.get("ppl", -1) for i in log_ls]
               for element in sublist
               for subelement in element]
    except:
        try:
            ppl = [element for
                   sublist in [i.get("ppl", -1) for i in log_ls]
                   for element in sublist]
        except:
            ppl = [i.get("ppl", -1) for i in log_ls]
    ppl = sum(ppl)/len(ppl)

    all_dic = {
        "all_generate_time":  f"all_generate_time: {sum([i.get('all_generate_time',0) for i in log_ls])}",
        "net_generate_time": f"net_generate_time: {sum([i.get('net_generate_time',0) for i in log_ls])}",
        "ppl": ppl,
        **log_dict
    }
    for i in all_dic.items():
        i = str(i)
        write_to_log(i)
        print(i)


def remove_prefix(prefix, main_string):
    if isinstance(main_string, list) and isinstance(prefix, list):
        return [remove_prefix(p, s) for p, s in zip(prefix, main_string)]
    elif isinstance(main_string, list):
        return [remove_prefix(prefix, mains) for mains in main_string]
    if isinstance(main_string, dict):
        main_string = main_string['content']
    if main_string.startswith(prefix):
        main_string = main_string[len(prefix):]
    return main_string


def write_to_log(data):
    log_filename = config.get('log_filename')
    if log_filename is not None:
        try:
            with open(log_filename, 'a') as log_file:
                log_file.write(str(data) + "\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
    else:
        print("Log file not created. Cannot write data.")



def preprocessing():
    print("GPU: " + str(torch.cuda.device_count()))
    create_log_file()


class QueryDataset(Dataset):
    def __init__(self, source_path, **kwargs):
        super().__init__()

        self.source_path = source_path
        self.size = kwargs.get('size', None)
        self.datasetname = "QueryDataset"

        self.tgt_file_path = self.get_next_dataset_file_path()
        write_to_log("data write to" + self.tgt_file_path)
        print(self.tgt_file_path)
        os.makedirs(os.path.dirname(self.tgt_file_path), exist_ok=True)
        if not os.path.exists(self.tgt_file_path):
            with open(self.tgt_file_path, 'w', encoding='utf-8') as file:
                json.dump([], file)

        meta_file = self.source_path + "metadata.json"
        data_file = self.source_path + "all.json"
        
        with open(meta_file, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)
        with open(data_file, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        if self.size is not None:
            self.data = self.data[:self.size]

        write_to_log(f"meta_data = {meta_file}")
        write_to_log(f"data_file = {data_file}")
        write_to_log(
            f"size = {self.size if self.size is not None else len(self.data)}")


    def write_data_to_tgt(self):
        print(self.tgt_file_path)
        new_data = [{key: value for key, value in original_dict.items() if (key != 'few_shot' and key != 'query_ls')}
                    for original_dict in self.data]
        with open(self.tgt_file_path, "w", encoding="utf-8") as json_file:
            json.dump(new_data, json_file, indent=4)

    def get_next_dataset_file_path(self):
        base_path = './result_data'
        index = 1
        while True:
            file_name = f'{self.datasetname}{index}.json'
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                break
            index += 1
        return file_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item['index'] = index + 1
        meta_data = self.metadata[item.get("relation","P6")]
        item['few_shot'] = meta_data['few_shot']
        ans = "-1"
        query_ls = {
            types: {}
            for types in config['types']
        }

        few_shot_specal_count = {
            "choose":1            
        }

        for query_type in config['types']:
            if query_type in few_shot_specal_count.keys():
                few_shot_count = few_shot_specal_count[query_type]
            else:
                few_shot_count = 3
            if query_type == "completion" or query_type == 'qa':
                query = "Q:" + \
                    item[f'{query_type}_query'].format(item['subjectLabel'])
                few_shot = [
                    f"({i['subjectLabel']}, {meta_data['DBpediaLabel']}, {i['objectLabel']})\n"+
                    "Q:" +
                    item[f'{query_type}_query'].format(i['subjectLabel'])
                    for i in item['few_shot']
                ]
                few_shot_ans = [
                    "A:" + i['objectLabel']
                    for i in item['few_shot']
                ]
            elif query_type == "fill":
                query = "Q:" + item[f'{query_type}_query'].format(
                    subject=item['subjectLabel'],
                    object='()',
                )
                few_shot = [
                    f"({i['subjectLabel']}, {meta_data['DBpediaLabel']}, {i['objectLabel']})\n"+
                    "Q:" + item[f'{query_type}_query'].format(
                        subject=i['subjectLabel'],
                        object='()',
                    )
                    for i in item['few_shot']
                ]
                few_shot_ans = [
                    "A:" + i['objectLabel']
                    for i in item['few_shot']
                ]
            elif query_type == "choose":
                random_num = random.random()
                non_ls = [i for i in item['few_shot']
                     if i['objectLabel']!= item['objectLabel']]
                
                if len(non_ls)<6:
                    few_shot_ls = random.sample(item['few_shot'],6)
                else:
                    few_shot_ls = random.sample(
                    non_ls,6
                )
                query = "Question: " + item["qa" + "_query"].format(item['subjectLabel']) +\
                    f"\noption: A:{item['objectLabel'] if random_num < 0.3 else few_shot_ls[3]['objectLabel']} " +\
                    f"B:{item['objectLabel'] if 0.3 <= random_num < 0.6 else few_shot_ls[4]['objectLabel']} " +\
                    f"C:{item['objectLabel'] if 0.6 <= random_num else few_shot_ls[5]['objectLabel']}" +\
                    "\nAnswer:"

                few_shot = [
                    f"({few_shot_ls[0]['subjectLabel']}, {meta_data['DBpediaLabel']}, {few_shot_ls[0]['objectLabel']})\n"+"Question: " + item["qa" + "_query"].format(few_shot_ls[j]['subjectLabel']) +
                    f"\noption:A:{few_shot_ls[0]['objectLabel']}"
                    f" B:{few_shot_ls[1]['objectLabel']} "
                    f"C:{few_shot_ls[2]['objectLabel']}"
                    for j in range(len(few_shot_ls))
                ][:1]

                few_shot_ans = [
                    "Answer:" + i['objectLabel']
                    for i in few_shot_ls
                ][:1]
            elif query_type == "fc":
                is_c = random.choice([0, 1])
                is_s_few = [
                    random.choice([0, 1])
                    for i in range(len(item['few_shot']))
                ]

                query = f"({item['subjectLabel']}, {meta_data['DBpediaLabel']}, {item['objectLabel']})\n"+"Statement:" + \
                    item[f'completion'+'_query'].format(item['subjectLabel']) + " " + [
                        random.choice(
                            item['few_shot']
                        )['subjectLabel'],
                        item['objectLabel']
                    ][is_c]
                few_shot = [
                    "Statement:" +
                    item["completion" + "_query"].format(item['few_shot'][j]['subjectLabel'])+" " +
                        [
                        item['few_shot'][(j+1) %
                                         len(item['few_shot'])]['objectLabel'],
                        item['few_shot'][j]['objectLabel']
                    ][is_s_few[j]]
                    for j in range(len(item['few_shot']))
                ]
                few_shot_ans = [
                    "A:" + ["False", "True"][is_s_few[j]]
                    for j in range(len(item['few_shot']))
                ]

                ans = ["0", "1"][is_c]
            elif query_type == "local":
                query = "Q:" + \
                    item[f'qa_query'].format(item['localitysubjectLabel'])
                few_shot = [
                    "Q:" +
                    item[f'qa_query'].format(i['subjectLabel'])
                    for i in item['few_shot']
                ]
                few_shot_ans = [
                    "A:" + i['objectLabel']
                    for i in item['few_shot']
                ]
                item['answers'] = "<ANS_SPLIT>".join(item['answers'])

            query_ls[query_type] = {
                "query": query,
                "few_shot": few_shot[:few_shot_count],
                "few_shot_ans": few_shot_ans[:few_shot_count],
                "ans": ans
            }
        item['query_ls'] = query_ls
        item['model_outputs'] = {}

        return item
def generate_for_kl_batch(dataset, model, tok, sta_idx,batch_size):

    item_ = [dataset[i] for i in range(sta_idx,sta_idx+batch_size)]

    to_test_input = [
            item['qa_query'].format(item['subjectLabel']) +" "+item['objectLabel']
            for item in item_
        ]
    if model.output_type == "edited":
        to_test_input = model.retrieval_and_merge(
            to_test_input,
            given_s = [
                i['subjectLabel'] for i in item_
            ]
        )
    prompt_tok = tok(
        to_test_input,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prefix_len = [len(tok(i)["input_ids"]) for i in to_test_input]

    model =model.model
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    
    prob_dir = dataset.tgt_file_path.replace(".json","")+"/prob_dir"
    if not os.path.exists(prob_dir):
        os.makedirs(prob_dir)
    for i in range(batch_size):
        prob_dist = - torch.nn.functional.log_softmax(logits[i, prefix_len[i] - 1, :], dim=0)
        name_from_model_name = config['model_name'][0].split("/")[-1].replace('-','_')
        fn = f"{name_from_model_name}_log_probs_{sta_idx+i}.pt"
        torch.save(prob_dist,  prob_dir +"/"+ fn)

def get_model_and_tokenizer(model_name):
    print("loading ", model_name)
    write_to_log("loading " + model_name)
    model_path = "/data/lzeng/editllm_dataset/result_1/model/" + model_name
    device = config['device']
    model, tok = None, None
    gc.collect()
    model_name = model_name.lower()
    if "edit" in model_name:
        output_type = config['model_type']
        write_to_log(f"output_type: {output_type}")
        print(f"output_type: {output_type}")
        
        knowledge_base = "/data/lzeng/editllm_dataset/FAME/data/demo_kg.ttl"
        model_name = model_name[len("edit_"):]
        model_path = "/data/lzeng/huggingface_models/model/" + model_name
        if 'llama' in model_name.lower():
            knowledge_base = knowledge_base
            model = skeme.memory_model(
                model_name, model_name,
                knowledge_base,
                "wikidata", None,
                output_type=output_type
            )
            write_to_log("knowledge_base: "+knowledge_base)
            tok = LlamaTokenizer.from_pretrained(
                model_path, padding_side='left'
            )
            tok.pad_token_id = tok.eos_token_id
        elif 'gpt' in model_name.lower():
            knowledge_base = knowledge_base
            model = skeme.memory_model(
                model_name, model_name,
                knowledge_base,
                "wikidata", None,
                output_type=output_type
            )
            write_to_log("knowledge_base: "+knowledge_base)
            tok = GPT2Tokenizer.from_pretrained(
                model_path, padding_side='left'
            )
            tok.pad_token_id = tok.eos_token_id

    else:

        if 't5' in model_name.lower():
            model = T5ForConditionalGeneration.from_pretrained(
                model_path, device_map='auto', local_files_only=True)
            tok = T5Tokenizer.from_pretrained(model_path, model_max_length=512)
        elif 'gpt' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='auto', local_files_only=True)
            tok = GPT2Tokenizer.from_pretrained(
                model_path, padding_side='left')
            tok.pad_token_id = tok.eos_token_id
        elif 'llama' in model_name.lower():
            model = LlamaForCausalLM.from_pretrained(
                model_path, device_map='auto', local_files_only=True)
            tok = LlamaTokenizer.from_pretrained(
                model_path, padding_side='left')
            tok.pad_token_id = tok.eos_token_id
        elif 'alpaca' in model_name.lower():
            model = LlamaForCausalLM.from_pretrained(
                model_path, device_map='auto', local_files_only=True)
            tok = LlamaTokenizer.from_pretrained(
                model_path, padding_side='left')
            tok.pad_token_id = tok.eos_token_id
        elif 'bloom' in model_name.lower():
            tok = BloomTokenizerFast.from_pretrained(
                model_path, device_map='auto')
            model = BloomForCausalLM.from_pretrained(model_path)
        else:
            raise NotImplementedError
    torch.cuda.empty_cache()
    print("loaded ", model_name)
    write_to_log("loaded " + model_name)
    return model, tok


def get_data():
    data = QueryDataset(
        "/data/lzeng/editllm_dataset/FAME/data/",
        # size = 1500
        )
    dataloader = DataLoader(
        dataset=data, batch_size=config['batch_size'], shuffle=False)
    config['output_file'] = data.tgt_file_path
    write_to_log(f"writting to {config['output_file']}")
    return dataloader



def pretty_print_dict(d, indent=4):
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key) + ":")
            pretty_print_dict(value, indent + 4)
        else:
            print(" " * indent + str(key) + ": " + str(value))


def merge_prompt_few_shot(sys_prompt, prompt, few_shot, ans, model_name):
    model_name = model_name.lower()
    few_shot = [[few_shot[i][j]
                 for i in range(len(few_shot))] for j in range(len(few_shot[0]))]
    ans = [[ans[i][j] for i in range(len(ans))] for j in range(len(ans[0]))]

    if "alpaca" == model_name:
        batch_prompts = []
        for prompts, answers, real_prompt in zip(few_shot, ans, prompt):
            batch_prompt = f"""{config['alpaca_prompt'].format(sys_prompt = sys_prompt)}"""
            for prompt, answer in zip(prompts, answers):
                _input = config['alpaca_prompt_few_shot'].format(
                    query=prompt,
                    ans=answer
                )
                batch_prompt += _input
            batch_prompt += f"\n### Input:\n{real_prompt}\n\n### Response:"
            batch_prompts.append(batch_prompt)
        return batch_prompts

    elif "llama2" in model_name and "c" in model_name:
        batch_prompts = []
        for prompts, answers, real_prompt in zip(few_shot, ans, prompt):
            batch_prompt = []

            system_instruction = {
                "role": "system",
                "content": f"You are a helpful, respectful and honest assistant. {sys_prompt}",
            }
            batch_prompt.append(system_instruction)

            for prompt, answer in zip(prompts, answers):
                user_prompt = {"role": "user", "content": prompt}
                assistant_response = {"role": "assistant", "content": answer}
                batch_prompt.extend([user_prompt, assistant_response])
            batch_prompt.append({"role": "user", "content": f"{real_prompt}"})

            batch_prompts.append(batch_prompt)
        return batch_prompts

    else:
        few_shot = [
            [x + '\n' + y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(few_shot, ans)
        ]
        if not config['types'] == ["choose"]:

            return [sys_prompt + '\n' + "\n".join(few_shot[i]) + '\n' + prompt[i]+"\nA:" for i in range(len(few_shot))]
        else:

            return [sys_prompt + '\n' + "\n".join(few_shot[i]) + '\n' + prompt[i] for i in range(len(few_shot))]


temp_dict = {}


def generate_one_output(entry, metadata, model, tok, name):
    type_ls = config['types']
    ret = {}
    log_info = {}
    inp_ls = {}
    ans_ls = {}
    for types in config['types']:

        with torch.no_grad():

            query_ls = entry['query_ls'][types]

            all_input = merge_prompt_few_shot(
                sys_prompt=config['sys_prompt'][types],
                prompt=query_ls['query'],
                few_shot=query_ls['few_shot'],
                ans=query_ls['few_shot_ans'],
                model_name=name
            )
            p_tok = tok(all_input, padding=True,
                        return_tensors="pt").to(config['device'])
            given_s = entry['subjectLabel'] 
            log_info_, logits, inp = model.generate(
                input_ids=p_tok['input_ids'],
                attention_mask=p_tok['attention_mask'],
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                given_s=given_s,
                task = types
            )
            toked_output = logits
            ret[types] = toked_output
            inp_ls[types] = inp
            ans_ls[types] = query_ls['ans']

            for k, v in log_info_.items():
                if k in log_info.keys():
                    log_info[k] += v
                else:
                    log_info[k] = v
    return log_info, ret, inp_ls, ans_ls



def output(name, model, tok, data):
    print('start output ' + name)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_to_log('start output ' + name + " at " + current_time)

    log_ls = []
    log_dict = {}

    with torch.no_grad():
        for i, entry in enumerate(tqdm(data)):

            if i % 1000 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"iter {i} at {current_time}")
            log_info, toked_output, inp, ans = generate_one_output(
                entry, data.dataset.metadata, model, tok, name
            )
            # for eval nkl
            # generate_for_kl_batch(data.dataset,model,tok,i*config['batch_size'],len(entry['index']))
            log_ls.append(log_info)
            for idx, id in enumerate(entry['index']):
                ret_output = {name: {}}
                for output_type in config['types']:

                    ret_output[name][output_type] = {
                        "content": toked_output[output_type][idx],
                        "inp": inp[output_type][idx],
                        "is_fact": ans[output_type][idx]
                    }
                try:
                    data.dataset.data[id -
                                      1]['model_outputs'].update(ret_output)
                except KeyError:
                    data.dataset.data[id - 1]['model_outputs'] = ret_output


    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_to_log('end  output ' + name + " at " + current_time)
    data.dataset.write_data_to_tgt()
    log_dict["memory_alloc_max"] = sum(
        [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    log_dict["memory_res_max"] = sum(
        [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())])
    log_dict["devices"] = int(torch.cuda.device_count())
    treat_log_info(log_ls, log_dict)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model name',default="edit_llama-2-7b-hf")
    parser.add_argument('--task', type=str, help='task',default="qa")

    args = parser.parse_args()

    if args.model is None or args.task is None:
        raise NotImplementedError
    else:
        print(args.model)
        print(args.task)
        config['model_name'] = [args.model]
        config['types'] = [args.task]
    return args



def main():
    create_log_file()
    data = get_data()
    parse_arguments()

    for name in config['model_name']:

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        model, tok = get_model_and_tokenizer(name)
        output(name, model, tok, data)
    log_info_txt = f"""
    model_name = {config["model_name"]}
    model_type = {config["model_type"]}
    task_type = {config['types']}
    """
    write_to_log(log_info_txt)
    print(log_info_txt)


    print()

print("GPU: " + str(torch.cuda.device_count()))
main()
