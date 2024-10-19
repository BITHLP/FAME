import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForCausalLM, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    AutoModel, AutoTokenizer, BertForQuestionAnswering,
    BloomForCausalLM, BloomTokenizerFast, DistilBertTokenizer, DistilBertModel
)
import threading
import typing
import json
from tqdm import tqdm
from datetime import datetime
import gc
import rdflib
import logging
import spacy
import time
import os
import getData
import utils

class memory_model(nn.Module):
    def __init__(self,
                 model,
                 tokenizer,
                 knowledge=None,
                 large_source=None,
                 nlp_utils_model_name=None,
                 output_type="edited",
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.output_type = output_type
        print(self.output_type)
        

        if model is None:
            self.model = None
            self.tokenizer = None
        elif isinstance(model, str):
            model_name = model
            model_path = "/data/lzeng/huggingface_models/model/" + model_name

            if 't5' in model_name.lower():
                model = T5ForConditionalGeneration.from_pretrained(
                    model_path, device_map='auto', local_files_only=True)
                tok = T5Tokenizer.from_pretrained(
                    model_path, model_max_length=512)
            elif 'gpt' in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map='auto', local_files_only=True)
                # model = None
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
            self.model = model
            self.tokenizer = tok

        else:
            self.model = model
            self.tokenizer = tokenizer
        self.memory = rdflib.Graph()
        self.large_source = large_source

        if self.output_type ==  "edited":
            
            if knowledge is not None:
                getData.load_ttl(self.memory, knowledge)


        self.sim_model = AutoModel.from_pretrained(
            "/data/lzeng/huggingface_models/model/contriever-msmarco").cuda()
        self.sim_tokenizer = AutoTokenizer.from_pretrained(
            "/data/lzeng/huggingface_models/model/contriever-msmarco")
        
        self.emd = None

    def save_knowledge_data(self, path=None):
        path_dir = "./knowledge_store"
        os.makedirs(path_dir, exist_ok=True)

        current_time = datetime.now()
        time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        print(time_string)
        if path is None:
            path = f'{path_dir}/{time_string}.ttl'
        with open(path, "w") as knowledge_file:
            knowledge_file.write(self.memory.serialize(format="turtle"))
        return path

    def extract_query(self, _input):
        def _extract_query(input_):
            try:
                temp_input = input_.split("\n")[-2]
                temp_input = utils.remove_prefix("Q:", temp_input)
            except BaseException:
                temp_input = input_
            return temp_input
        _input = _extract_query(_input)
        return _input
    def query_triplet(self, item, kg=None, position=0):
        if kg is None:
            return getData.query_triplet(self.memory, item, position)
        else:
            return getData.query_triplet(kg, item, position)

    def _sort_by_relate(self, triplet_ls, _input, top_k=1):
        if (len(triplet_ls)) == 0:
            return triplet_ls
        if (len(triplet_ls[0])) == 0:
            return triplet_ls


        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(
                dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        def get_sent_embeddings(sents, contriever, tok, BSZ=32):
            all_embs = []
            for i in (range(0, len(sents), BSZ)):
                sent_batch = sents[i:i + BSZ]
                inputs = tok(sent_batch, padding=True,
                             truncation=True, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = contriever(**inputs)
                    embeddings = mean_pooling(
                        outputs[0], inputs['attention_mask'])
                all_embs.append(embeddings.cpu())
            all_embs = torch.vstack(all_embs)
            return all_embs

        def retrieve_facts(query, fact_embs, contriever, tok, k=5):
            inputs = tok([query], padding=True, truncation=True,
                         return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = contriever(**inputs)
                query_emb = mean_pooling(
                    outputs[0], inputs['attention_mask']).cpu()
            sim = (query_emb @ fact_embs.T)[0]
            knn = sim.topk(min(k, fact_embs.shape[0]), largest=True)
            return knn.indices

        to_query = [f"{i[0]} {i[1]}" for i in triplet_ls]
        embs = get_sent_embeddings(
            to_query, self.sim_model, self.sim_tokenizer)
        fact_ids = retrieve_facts(
            _input, embs, self.sim_model, self.sim_tokenizer, k=top_k)
        return [triplet_ls[i] for i in fact_ids]

    def sort_by_relate(self, triplet_ls, _input, top_k=1):
        time_before_sort = time.time()
        res = self._sort_by_relate(triplet_ls, _input, top_k=top_k)
        time_after_sort = time.time()
        return res

    def insert_knowledeg(self, _input, gt=None):
        s, r = self.extract_sr(_input)
        self._insert_knowledeg(s)
        if gt is not None:
            print(s, r, gt, _input)
            to_add = [[_s, _input, gt]
                      for _s in s] if isinstance(s, list) else [s, r, gt]
            self._add_to_memory(to_add)
        return s

    def batch_insert_konwledeg(self, _input):
        if isinstance(_input, str):
            _input = [_input]
        start_time = time.time()
        for s in _input:
            s, _ = self.extract_sr(_input)
            self._insert_knowledeg(s)
        end_time = time.time()


    def merge_knowledge_and_query(self, knowledge, _input):

        if len(knowledge) == 0:
            return _input
        if len(knowledge[0]) == 0:
            return _input
        assert len(knowledge[0]) == 3, \
            f"knowledge should in format [[subject,relation,object],...],get {knowledge[0]}"

        def encode_knowledge_to_str(knowledge):
            return [f"({sublist[0]+', '+sublist[1]}, {sublist[2]})" for sublist in knowledge]
        # print(knowledge)
        knowledge_to_insert = encode_knowledge_to_str(knowledge)
        knowledge_to_insert = "\n".join(knowledge_to_insert)


        lines = _input.splitlines()
        lines.insert(-2, knowledge_to_insert)
        new_text = '\n'.join(lines)

        return new_text
    
    def merge_knowledge_and_query_for_fc(self, knowledge, _input):

        if len(knowledge) == 0:
            return _input
        if len(knowledge[0]) == 0:
            return _input
        assert len(knowledge[0]) == 3, \
            f"knowledge should in format [[subject,relation,object],...],get {knowledge[0]}"

        def encode_knowledge_to_str(knowledge):
            return [f"({sublist[0]+', '+sublist[1]}, {sublist[2]})" for sublist in knowledge]

        knowledge_to_insert = encode_knowledge_to_str(knowledge)
        knowledge_to_insert = "\n".join(knowledge_to_insert) 



        lines = _input.splitlines()
        lines.insert(-2, knowledge_to_insert)

        new_text = '\n'.join(lines)

        return new_text
    
    def merge_knowledge_and_query_for_choose(self, knowledge, _input):

        if len(knowledge) == 0:
            return _input
        if len(knowledge[0]) == 0:
            return _input
        assert len(knowledge[0]) == 3, \
            f"knowledge should in format [[subject,relation,object],...],get {knowledge[0]}"

        def encode_knowledge_to_str(knowledge):
            return [f"({sublist[0]+', '+sublist[1]}, {sublist[2]})" for sublist in knowledge]

        knowledge_to_insert = encode_knowledge_to_str(knowledge)
        knowledge_to_insert = "\n".join(knowledge_to_insert)



        lines = _input.splitlines()
        lines.insert(-3, knowledge_to_insert)

        new_text = '\n'.join(lines)

        return new_text

    def merge_knowledge_and_query_for_completion(self, knowledge, _input):

        if len(knowledge) == 0:
            return _input
        if len(knowledge[0]) == 0:
            return _input
        assert len(knowledge[0]) == 3, \
            f"knowledge should in format [[subject,relation,object],...],get {knowledge[0]}"

        def encode_knowledge_to_str(knowledge):
            return [f"{sublist[0]+','+sublist[1]}, {sublist[2]}\n" for sublist in knowledge]

        knowledge_to_insert = encode_knowledge_to_str(knowledge)
        knowledge_to_insert = "These (subject, relationship, object) triplets may useful\n" + \
            "\n".join(knowledge_to_insert)


        lines = _input.splitlines()
        lines.insert(-1, knowledge_to_insert)
        new_text = '\n'.join(lines)

        return new_text

    def _add_to_memory(self, ls):
        def flatten_nested_list(nested_list):
            flat_list = []

            for item in nested_list:
                if isinstance(item, list):
                    flat_list.extend(flatten_nested_list(item))
                else:
                    flat_list.append(item)

            return flat_list

        def reshape_list(lst, shape):
            if len(shape) == 1:
                return lst
            else:
                inner_shape = shape[1:]
                n = shape[0]
                return [reshape_list(lst[i:i + n], inner_shape) for i in range(0, len(lst), n)]

        flat_list = flatten_nested_list(ls)
        reshaped_list = reshape_list(flat_list, [3, len(flat_list) // 3])
        for i in reshaped_list:
            s, r, o = i
            s = rdflib.Literal(s)
            r = rdflib.Literal(r)
            o = rdflib.Literal(o)
            self.memory.set((s, r, o))

    def _insert_knowledeg(self, item ):
        if self.large_source is None:
            pass
        
        s_appear_as_s = self.query_triplet(
            item, kg=self.large_source, position=0)
        s_appear_as_o = [[]]
        s_appear = s_appear_as_s + s_appear_as_o
        self._add_to_memory(s_appear)
        return s_appear

    def query_knowledge(self, _input, knowledge_count=1, given_s=None):
        if isinstance(_input, list):
            return [
                self.query_knowledge(i, knowledge_count, g) for i, g in zip(_input, given_s)
            ]       
     

        if given_s is not None:
            s = given_s
        else:
            s, r = self.extract_sr(_input)

        s_appear_as_s = self.query_triplet(s, position=0)
        s_appear_as_o = [[]]

        if s_appear_as_s is None:
            s_appear_as_s = [[]]
        if s_appear_as_o is None:
            s_appear_as_o = [[]]
        s_appear_as_s.extend(s_appear_as_o)
        s_appear = [i for i in s_appear_as_s if i != []]
        s_related = self.sort_by_relate(
            s_appear, _input, top_k=knowledge_count)
        if (len(s_related) > 0):
            if (len(s_related[0]) > 0):
                pass
        else:
            s_related = self._insert_knowledeg(s)

        return s_related[:knowledge_count]

    def _generate(self, input_text, *args, **kwargs):
        unused_para = ["knowledge_count", "given_s"]
        for i in unused_para:
            kwargs[i] = None
        new_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        if hasattr(self.model, "name_or_path") \
                and "t5" in self.model.name_or_path.lower():
            kwargs["input_ids"] = new_input["input_ids"]
            kwargs["attention_mask"] = new_input["attention_mask"]
        else:
            kwargs["input_ids"] = new_input["input_ids"]
            kwargs["attention_mask"] = new_input["attention_mask"]
        with torch.no_grad():
            res = self.model.generate(*args, **kwargs)

        return res
    
    def retrieval_and_merge(self,input_text, *args, **kwargs):
        related_knowledge = self.query_knowledge(
            input_text,
            knowledge_count=kwargs.get("knowledge_count", 1),
            given_s=kwargs.get("given_s", None),
        )
        input_text = [
            self.merge_knowledge_and_query_for_choose(_related_knowledge, _input_text)
            for _related_knowledge, _input_text
            in zip(related_knowledge, input_text)
        ]
        if self.show_example_input_flag == 1:
            self.show_example_input_flag = 2
            to_print = input_text[0] \
                if isinstance(input_text, list) \
                else input_text
            print(to_print)

        return input_text

    def generate(self, *args, **kwargs):

        log_info = {}
        time_sta = time.time()
        input_text = self.tokenizer.batch_decode(
            kwargs["input_ids"], skip_special_tokens=True)

        assert len(args) == 0, "Should only pass named arguments to generate()"

        if self.output_type == "raw":
            kwargs["task"] = None
            time_before_generate = time.time()
            ret = self._generate(input_text, *args, **kwargs)
            time_after_generate = time.time()
            toked_output = self.tokenizer.batch_decode(
                ret, skip_special_tokens=True)
            re_toked_output = utils.remove_prefix_with_len(
                input_text, toked_output)  # remove query here
            time_end = time.time()
            log_info.update(
                {
                    "all_generate_time": time_end-time_sta,
                    "net_generate_time": time_after_generate-time_before_generate,
                }
            )
            return log_info, re_toked_output, input_text

        related_knowledge = self.query_knowledge(
            input_text,
            knowledge_count=kwargs.get("knowledge_count", 1),
            given_s=kwargs.get("given_s", None),
        )
        task = kwargs.get("task", None)
        kwargs["knowledge_count"] = None
        kwargs["given_s"] = None
        kwargs["task"] = None


        if task == "choose":
            input_text = [
                self.merge_knowledge_and_query_for_choose(_related_knowledge, _input_text)
                for _related_knowledge, _input_text
                in zip(related_knowledge, input_text)
            ]
        elif task == "fc":
            input_text = [
                self.merge_knowledge_and_query_for_fc(_related_knowledge, _input_text)
                for _related_knowledge, _input_text
                in zip(related_knowledge, input_text)
            ]
        else:
            input_text = [
                self.merge_knowledge_and_query(_related_knowledge, _input_text)
                for _related_knowledge, _input_text
                in zip(related_knowledge, input_text)
            ]

        time_before_generate = time.time()
        ret = self._generate(input_text, *args, **kwargs)
        time_after_generate = time.time()

        toked_output = self.tokenizer.batch_decode(
            ret, skip_special_tokens=True)
        re_toked_output = utils.remove_prefix_with_len(
            input_text, toked_output)  
        time_end = time.time()
        log_info.update(
            {
                "all_generate_time": time_end-time_sta,
                "net_generate_time": time_after_generate-time_before_generate,
            }
        )
        return log_info, re_toked_output, input_text
