# -*- coding=utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2025-06-01
# @Contact: slinzhai@gmail.com

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

import sys
sys.path.append('./')
import json
import torch
import typing
import argparse
import random
import time
from itertools import chain
import scipy
from pathlib import Path
import numpy as np
from random import sample
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)

from plms.gptj import GPTJForCausalLM
from plms.llama2 import LlamaForCausalLM as Llama2ForCausalLM
from plms.llama3 import LlamaForCausalLM as Llama3ForCausalLM


CONFIG = {
    "name": "Parameter-Aware Contrastive Knowledge Editing",
    "gptj":{
        "path": "/home/zhaisonglin/disk/songlin/resources/GPTJ/",
        "bias": True,
        "ffn1_nm_key": "mlp.fc_in",
        "ffn2_nm_key": "mlp.fc_out",
        "ffn1_nm_wt": "transformer.h.{}.mlp.fc_in.weight",
        "ffn2_nm_wt": "transformer.h.{}.mlp.fc_out.weight",
        "ffn1_nm_bias": "transformer.h.{}.mlp.fc_in.bias",
        "ffn2_nm_bias": "transformer.h.{}.mlp.fc_out.bias",
        "ffn1_grad": "self.model.transformer.h[{}].mlp.fc_in.weight.grad.detach()",
        "ffn2_grad": "self.model.transformer.h[{}].mlp.fc_out.weight.grad.detach()",
        "lm_embed_device": "self.model.transformer.wte.weight.device",
        "lm_head_device": "self.model.transformer.ln_f.weight.device"
    },
    "llama2-7B":{
        "path": "/home/zhaisonglin/disk/songlin/resources/Llama2/7B/",
        "bias": False,
        "ffn1_nm_key": "mlp.up_proj",
        "ffn2_nm_key": "mlp.down_proj",
        "ffn1_nm_wt": "model.layers.{}.mlp.up_proj.weight",
        "ffn2_nm_wt": "model.layers.{}.mlp.down_proj.weight",
        "ffn1_grad": "self.model.model.layers[{}].mlp.up_proj.weight.grad.detach()",
        "ffn2_grad": "self.model.model.layers[{}].mlp.down_proj.weight.grad.detach()",
        "lm_embed_device": "self.model.model.embed_tokens.weight.device",
        "lm_head_device": "self.model.lm_head.weight.device"
    },
    "llama3-8B":{
        "path": "/home/zhaisonglin/disk/songlin/resources/Llama3/8B",
        "bias": False,
        "ffn1_nm_key": "mlp.up_proj",
        "ffn2_nm_key": "mlp.down_proj",
        "ffn1_nm_wt": "model.layers.{}.mlp.up_proj.weight",
        "ffn2_nm_wt": "model.layers.{}.mlp.down_proj.weight",
        "ffn1_grad": "self.model.model.layers[{}].mlp.up_proj.weight.grad.detach()",
        "ffn2_grad": "self.model.model.layers[{}].mlp.down_proj.weight.grad.detach()",
        "lm_embed_device": "self.model.model.embed_tokens.weight.device",
        "lm_head_device": "self.model.lm_head.weight.device"
    },
    "data":{
        "zsre": "./data/zsre_edit{}.json",
        "counterfact": "./data/counterfact_edit{}.json"
    },
    "optimizer":{ "lr": 1e-5, "weight_decay": 0. }
}

SEGMENTS = [[{'start':   0, 'end':1000}, {'start':2800, 'end':3800}, {'start':4800, 'end':5800}]]

def z_score_norm(mx:torch.tensor):
    assert mx.dim() == 2, 'Error in Dimension!'
    mean = torch.unsqueeze(mx.mean(1), dim=-1)
    std = torch.unsqueeze(mx.std(1), dim=-1)
    return (mx-mean)/std

def rand_array(size, lower_bound, upper_bound, exclude=[]):
    rand_list = list()
    while len(rand_list)<size:
        idx = random.randint(lower_bound, upper_bound)
        while (idx in rand_list) or (idx in exclude):
            idx = random.randint(lower_bound, upper_bound)
        rand_list.append(idx)
    return rand_list

def flatten(data:list): return list(chain.from_iterable(data))

def now(format='%Y-%m-%d-%H:%M:%S'): return time.strftime(format, time.localtime())


class ZsRE(Dataset):
    def __init__(self, data_fp: str) -> None:
        super().__init__()
        with open(data_fp, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return tuple([idx, self._data[idx]])
    
    def sample(self, size:int, random=False):
        if random: self._data = sample(self._data, k=size)
        else: self._data = self._data[:size]

    def segments(self, group_idx, seg_idx=None):
        if seg_idx is None:
            # Combine all segments data in a group, 3K
            data = list()
            for seg in SEGMENTS[group_idx]:
                data.extend(self._data[seg['start']:seg['end']])
            self._data = data
        else:
            # Just Load one setment, 1K
            self._data = self._data[SEGMENTS[group_idx][seg_idx]['start']:\
                                    SEGMENTS[group_idx][seg_idx]['end']]

    def reproduce(self, model, tokenizer, emb_device, write_fp=None):
        _data = self._data
        data = list()
        pbar = tqdm(total=len(_data), ncols=75)
        pbar.set_description_str(desc='Reproduce->')
        model.eval()
        with torch.no_grad():
            for record in _data:
                record_update = dict()
                record_update['case_id'] = record['case_id']
                inputs = list()
                inputs.append(record['prompt'] + ' ' + record['target_new'])
                inputs.append(record['paraphrase_prompts'] + ' ' + record['target_new'])
                inputs.append(record['locality_prompt'] + ' ' + record['locality_ground_truth'])
                inputs.extend([_+' '+record['target_new'] for _ in record['similar_prompts']])
                #
                inputs = tokenizer(inputs, padding=True, return_tensors='pt').to(emb_device)
                lengths = inputs.attention_mask.sum(1)
                preds = torch.argmax(model(**inputs).logits, dim=-1).detach().cpu().tolist()
                record_update['prompt'] = [record['prompt'], preds[0][:lengths[0]]]
                record_update['paraphrase_prompts'] = [record['paraphrase_prompts'], preds[1][:lengths[1]]]
                record_update['locality_prompt'] = [record['locality_prompt'], preds[2][:lengths[2]]]
                record_update['similar_prompts'] = list(map(lambda sim,x,y: [sim, x[:y]],\
                                                        record['similar_prompts'], preds[3:], lengths[3:]))
                record_update['target_new'] = record['target_new']
                record_update['ground_truth'] = record['ground_truth']
                record_update['locality_ground_truth'] = record['locality_ground_truth']
                record_update['cond'] = record['cond']
                #
                data.append(record_update)
                pbar.update(1)
        pbar.close()
        self._data = data
        if write_fp is not None:
            with open(write_fp, 'w', encoding='utf-8') as ofstream:
                json.dump(data, ofstream, indent=4)


class CounterFact(Dataset):
    def __init__(self, data_fp: str):
        with open(data_fp, "r", encoding='utf-8') as f:
            self._data = json.load(f)
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return tuple([idx, self._data[idx]])
    
    def sample(self, size:int, random=False):
        if random: self._data = sample(self._data, k=size)
        else: self._data = self._data[:size]

    def segments(self, group_idx, seg_idx=None):
        if seg_idx is None:
            # Combine all segments data in a group, 3K
            data = list()
            for seg in SEGMENTS[group_idx]:
                data.extend(self._data[seg['start']:seg['end']])
            self._data = data
        else:
            # Just Load one setment, 1K
            self._data = self._data[SEGMENTS[group_idx][seg_idx]['start']:\
                                    SEGMENTS[group_idx][seg_idx]['end']]
    
    def reproduce(self, model, tokenizer, emb_device, write_fp=None):
        _data = self._data
        data = list()
        pbar = tqdm(total=len(_data), ncols=75)
        pbar.set_description_str(desc='Reproduce->')
        with torch.no_grad():
            for record in _data:
                record_update = dict()
                record_update['case_id'] = record['case_id']
                inputs = list()
                inputs.append(record['prompt'] + ' ' + record['target_new'])
                inputs.append(record['rephrase_prompt'] + ' ' + record['target_new'])
                inputs.append(record['locality_prompt'] + ' ' + record['locality_ground_truth'])
                #
                inputs = tokenizer(inputs, padding=True, return_tensors='pt').to(emb_device)
                lengths = inputs['attention_mask'].sum(1)
                preds = torch.argmax(model(**inputs).logits, dim=-1).detach().cpu().tolist()
                record_update['prompt'] = [record['prompt'], preds[0][:lengths[0]]]
                record_update['rephrase_prompt'] = [record['rephrase_prompt'], preds[1][:lengths[1]]]
                record_update['locality_prompt'] = [record['locality_prompt'], preds[2][:lengths[2]]]
                record_update['target_new'] = record['target_new']
                record_update['ground_truth'] = record['ground_truth']
                record_update['locality_ground_truth'] = record['locality_ground_truth']
                #
                data.append(record_update)
                pbar.update(1)
        pbar.close()
        self._data = data
        if write_fp is not None:
            with open(write_fp, 'w', encoding='utf-8') as ofstream:
                json.dump(data, ofstream, indent=4)


class ZsREDataLoader(DataLoader):
    def __init__(self, dataset:ZsRE, batch_size, shuffle=True):
        super(ZsREDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,
            collate_fn=self.zsre_collate_fn, shuffle=shuffle
        )
    
    def zsre_collate_fn(self, batch):
        batch_idx = [b[0] for b in batch]
        case_ids = [b[1]["case_id"] for b in batch]
        #
        edit_prompts = [b[1]["prompt"][0] for b in batch]
        edit_ans = [b[1]["target_new"] for b in batch]
        edit_pred = [b[1]["prompt"][1] for b in batch]
        edit_inputs = list(map(lambda x, y: x+' '+y, edit_prompts, edit_ans))
        #
        rep_prompts = [b[1]["paraphrase_prompts"][0] for b in batch]
        rep_ans = edit_ans
        rep_pred = [b[1]["paraphrase_prompts"][1] for b in batch]
        rep_inputs = list(map(lambda x, y: x+' '+y, rep_prompts, rep_ans))
        #
        loc_prompts = [b[1]["locality_prompt"][0] for b in batch]
        loc_ans = [b[1]["locality_ground_truth"] for b in batch]
        loc_pred = [b[1]["locality_prompt"][1] for b in batch]
        loc_inputs = list(map(lambda x, y: x+' '+y, loc_prompts, loc_ans))
        return tuple([case_ids,
                      edit_inputs, edit_prompts, edit_ans, edit_pred,
                      rep_inputs, rep_prompts, rep_ans, rep_pred,
                      loc_inputs, loc_prompts, loc_ans, loc_pred])


class CounterFactDataLoader(DataLoader):
    def __init__(self, dataset:CounterFact, batch_size, shuffle=True):
        super(CounterFactDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,
            collate_fn=self.cf_collate_fn, shuffle=shuffle
        )
    
    def cf_collate_fn(self, batch):
        batch_idx = [b[0] for b in batch]
        #
        case_ids = [b[1]["case_id"] for b in batch]
        #
        edit_prompts = [b[1]["prompt"][0] for b in batch]
        edit_ans = [b[1]["target_new"] for b in batch]
        edit_pred = [b[1]["prompt"][1] for b in batch]
        edit_inputs = list(map(lambda x, y: x+' '+y, edit_prompts, edit_ans))
        #
        rep_prompts = [b[1]["rephrase_prompt"][0] for b in batch]
        rep_ans = edit_ans
        rep_pred = [b[1]["rephrase_prompt"][1] for b in batch]
        rep_inputs = list(map(lambda x, y: x+' '+y, rep_prompts, rep_ans))
        #
        loc_prompts = [b[1]["locality_prompt"][0] for b in batch]
        loc_ans = [b[1]["locality_ground_truth"] for b in batch]
        loc_pred = [b[1]["locality_prompt"][1] for b in batch]
        loc_inputs = list(map(lambda x, y: x+' '+y, loc_prompts, loc_ans))
        #
        return tuple([case_ids,
                      edit_inputs, edit_prompts, edit_ans, edit_pred,
                      rep_inputs, rep_prompts, rep_ans, rep_pred,
                      loc_inputs, loc_prompts, loc_ans, loc_pred])


class PAC(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        PAC_INPUTS_DOCSTRING=r'''
        Class of Parameter-Aware Contrastive Knowledge Editing model
        -----
        llm_name:
            the LLM name to edit, supporting:
            GPTJ,   GPT-J                           --->>>  perform editing on GPTJ-6B   model
            Llama2, Llama-2, Llama2-7B, Llama-2-7B  --->>>  perform editing on Llama2-7B model
            Llama3, Llama-3, Llama3-8B, Llama-3-8B  --->>>  perform editing on Llama3-8B model
        '''
        self.args = args
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.llm = None
        self.kv_idxes_pos = None
        self.kv_idxes_neg = None
        self.config = None
        #
        name = self.args.llm_name.lower()
        if name in ['gptj','gpt-j','gptj-6b','gpt-j-6b']:
            self.llm = 'gptj'
            self.model = GPTJForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG[self.llm]['path'])
        elif name in ['llama2','llama-2','llama2-7b','llama-2-7b']:
            self.llm = 'llama2-7B'
            self.model = Llama2ForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = LlamaTokenizer.from_pretrained(CONFIG[self.llm]['path'])
        elif name in ['llama3','llama-3','llama3-8b','llama-3-8b']:
            self.llm = 'llama3-8B'
            self.model = Llama3ForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG[self.llm]['path'])
        else: raise Exception(f'The current approach does not support the {llm_name}!\
                               \nSee{PAC_INPUTS_DOCSTRING}')
        # Set Attn layer to no_grad
        for name, param in self.model.named_parameters():
            if param.requires_grad and\
               CONFIG[self.llm]['ffn1_nm_key'] not in name and\
               CONFIG[self.llm]['ffn2_nm_key'] not in name:
                param.requires_grad = False
        # Setting the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.optimizer = torch.optim.Adam(self.model.parameters(),**CONFIG['optimizer'])
    
    def tokenize(self, batch_inputs:list, batch_prompts:list, batch_preds:list=None):
        inputs = self.tokenizer(batch_inputs, padding=True, return_tensors='pt')
        seq_length = inputs.attention_mask.sum(1).tolist()
        labels = inputs.input_ids.detach().clone()
        prompts_ids = self.tokenizer(batch_prompts).input_ids
        max_length = labels.size(1)
        prompts_length = list()
        original_preds = list()
        for idx, one_ids in enumerate(prompts_ids):
            # padding_side=Right:  -100... + pre_pred... + pad...
            labels[idx][:len(one_ids)] = -100
            labels[idx][seq_length[idx]:] = -100
            prompts_length.append(len(one_ids))
            if batch_preds is not None:
                one_pred_ = [-100]*max_length
                one_pred_[len(one_ids):seq_length[idx]] = batch_preds[idx][len(one_ids)-1:seq_length[idx]-1]
                original_preds.append(one_pred_)
        return {'input_ids':inputs.input_ids, 'attention_mask':inputs.attention_mask, 'labels':labels}, seq_length, prompts_length, torch.as_tensor(original_preds)

    def assessment(self, assess_loader, epoch_idx):
        # the assessment is carries out for specific epoches
        if epoch_idx%self.args.eta == 0:
            importance = dict()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    importance[name] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
            # Carry out the importance assessment
            pbar = tqdm(total=assess_loader.dataset.__len__(), ncols=75, leave=True)
            pbar.set_description_str(desc='Assess->')
            self.model.train()
            for _, batch in enumerate(assess_loader):
                data_crt, _, _, _ = self.tokenize(batch[1], batch[2])
                self.optimizer.zero_grad()
                loss = self.model(**data_crt).loss
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        parma_importance = param.grad.detach()
                        importance[name] = importance[name]+parma_importance\
                                           if importance[name].size()==parma_importance.size()\
                                           else importance[name]+parma_importance.T
                pbar.update(len(batch[0]))
            pbar.close()
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            self.model.eval()
            # Retreive the importance kv idxes
            kv_idxes_pos = dict()
            kv_idxes_neg = dict()
            importance_full_dict = dict()
            importance_crit_dict = dict()
            for layer in range(self.model.config.num_hidden_layers):
                ffn1_name = CONFIG[self.llm]['ffn1_nm_wt'].format(layer)
                ffn2_name = CONFIG[self.llm]['ffn2_nm_wt'].format(layer)
                ffn1_importance = importance[ffn1_name]
                ffn2_importance = importance[ffn2_name]
                # Transpose all importance matrix into [hidden_dim, num_feature]
                # which will facilitate subsequent operations
                if ffn1_importance.size(0) != self.model.config.hidden_size:
                    ffn1_importance = ffn1_importance.T
                if ffn2_importance.size(0) != self.model.config.hidden_size:
                    ffn2_importance = ffn2_importance.T
                ffn1_mean_importance = torch.abs(z_score_norm(ffn1_importance)).mean(0)
                ffn2_mean_importance = torch.abs(z_score_norm(ffn2_importance)).mean(0)
                importance_full_dict[f'Layer_{layer}_K'] = ffn1_mean_importance.clone().to('cpu').tolist()
                importance_full_dict[f'Layer_{layer}_V'] = ffn2_mean_importance.clone().to('cpu').tolist()
                ffn1_idx_pos = torch.argsort(ffn1_mean_importance, descending=True, stable=True)[:self.args.topk]
                ffn2_idx_pos = torch.argsort(ffn2_mean_importance, descending=True, stable=True)[:self.args.topk]
                ffn1_idx_neg = torch.argsort(ffn1_mean_importance, descending=False, stable=True)[:self.args.bottomk]
                ffn2_idx_neg = torch.argsort(ffn2_mean_importance, descending=False, stable=True)[:self.args.bottomk]
                if ffn1_name in kv_idxes_pos.keys(): raise Exception('Keyword repetition!')
                if ffn2_name in kv_idxes_pos.keys(): raise Exception('Keyword repetition!')
                importance_crit_dict[f'Layer_{layer}_K'] = ffn1_mean_importance[ffn1_idx_pos].clone().to('cpu').tolist()
                importance_crit_dict[f'Layer_{layer}_V'] = ffn2_mean_importance[ffn2_idx_pos].clone().to('cpu').tolist()
                kv_idxes_pos[ffn1_name] = ffn1_idx_pos
                kv_idxes_pos[ffn2_name] = ffn2_idx_pos
                kv_idxes_neg[ffn1_name] = ffn1_idx_neg
                kv_idxes_neg[ffn2_name] = ffn2_idx_neg
                if CONFIG[self.llm]['bias']:
                    kv_idxes_pos[CONFIG[self.llm]['ffn1_nm_bias'].format(layer)] = ffn1_idx_pos
                    kv_idxes_pos[CONFIG[self.llm]['ffn2_nm_bias'].format(layer)] = torch.arange(self.model.config.hidden_size)
                    kv_idxes_neg[CONFIG[self.llm]['ffn1_nm_bias'].format(layer)] = ffn1_idx_neg
                    kv_idxes_neg[CONFIG[self.llm]['ffn2_nm_bias'].format(layer)] = torch.arange(self.model.config.hidden_size)
            self.kv_idxes_pos = kv_idxes_pos
            self.kv_idxes_neg = kv_idxes_neg
            importance = None # Clear cuda memory
        torch.cuda.empty_cache()
        return self.kv_idxes_pos, self.kv_idxes_neg
    
    def edit(self, batch_inputs, batch_prompts, batch_preds):
        data_crt, _,_, batch_preds = self.tokenize(batch_inputs, batch_prompts, batch_preds)
        losses = list()
        self.optimizer.zero_grad()
        loss = self.model(**data_crt).loss
        loss.backward()
        losses.append(loss.item())
        if self.args.topk is None:
            # Finetuning all FFNs
            self.optimizer.step()
            torch.cuda.empty_cache()
            return losses
        # Optimize the topk KV path
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if self.args.topk == -1: continue
            current_idx = self.kv_idxes_pos[name]
            # Weight matrix of the first FFN: Require column-wise operation
            # @1: Backup gradient of required slices
            # @2: Set other slices to 0
            # @3: Restore the gradient of required slices
            if CONFIG[self.llm]['ffn1_nm_key'] in name:
                if 'weight' in name:
                    if param.size(0) == self.model.config.hidden_size:
                        grad_ = param.grad[:,current_idx]
                        param.grad[:] = 0.
                        param.grad[:,current_idx] = grad_
                    else:
                        grad_ = param.grad[current_idx]
                        param.grad[:] = 0.
                        param.grad[current_idx] = grad_
                elif 'bias' in name:
                    grad_ = param.grad[current_idx]
                    param.grad[:] = 0.
                    param.grad[current_idx] = grad_
                else: raise Exception(f'{name} cannot be handled!')
            # Weight matrix of the second FFN: Require row-wise operation
            elif CONFIG[self.llm]['ffn2_nm_key'] in name:
                if 'weight' in name:
                    if param.size(1) == self.model.config.hidden_size:
                        grad_ = param.grad[current_idx]
                        param.grad[:] = 0.
                        param.grad[current_idx] = grad_
                    else:
                        grad_ = param.grad[:,current_idx]
                        param.grad[:] = 0.
                        param.grad[:,current_idx] = grad_
                elif 'bias' in name:
                    pass # Anything to do, since this entire bias vector should be updated
                else: raise Exception(f'{name} cannot be handled!')
            else: raise Exception(f'{name} cannot be handled!')
            # param.grad = param.grad*(self.scale[name.split('.')[2]].to(param.grad.device))
        self.optimizer.step()
        torch.cuda.empty_cache()

        """ Contrastive Loss """
        if self.args.lamb == 0.:
            torch.cuda.empty_cache()
            return losses
        # if the lambda is set to 0,
        # indicating the contrastive loss is not used
        data_crt['labels'] = batch_preds
        self.optimizer.zero_grad()
        loss = self.model(**data_crt).loss*self.args.lamb
        loss.backward()
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if self.args.bottomk == 0: continue
            current_idx = self.kv_idxes_neg[name]
            if CONFIG[self.llm]['ffn1_nm_key'] in name:
                if 'weight' in name:
                    if param.size(0) == self.model.config.hidden_size:
                        grad_ = param.grad[:,current_idx]
                        param.grad[:] = 0.
                        param.grad[:,current_idx] = grad_
                    else:
                        grad_ = param.grad[current_idx]
                        param.grad[:] = 0.
                        param.grad[current_idx] = grad_
                elif 'bias' in name:
                    grad_ = param.grad[current_idx]
                    param.grad[:] = 0.
                    param.grad[current_idx] = grad_
                else: raise Exception(f'{name} cannot be handled!')
            # Weight matrix of the second FFN: Require row-wise operation
            elif CONFIG[self.llm]['ffn2_nm_key'] in name:
                if 'weight' in name:
                    if param.size(1) == self.model.config.hidden_size:
                        grad_ = param.grad[current_idx]
                        param.grad[:] = 0.
                        param.grad[current_idx] = grad_
                    else:
                        grad_ = param.grad[:,current_idx]
                        param.grad[:] = 0.
                        param.grad[:,current_idx] = grad_
                elif 'bias' in name:
                    pass # Anything to do, since this entire bias vector should be updated
                    grad_ = param.grad
                else: raise Exception(f'{name} cannot be handled!')
            else: raise Exception(f'{name} cannot be handled!')
        self.optimizer.step()
        return losses
    
    def forward(self, batch_inputs, batch_prompts, output_hidden_states=False):
        self.model.eval()
        inputs, _, _, _ = self.tokenize(batch_inputs, batch_prompts)
        del inputs['labels'] # To avoid loss computation for saving time
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
        return outputs

    @property
    def emb_device(self): return eval(CONFIG[self.llm]['lm_embed_device'])
    @property
    def lm_device(self): return eval(CONFIG[self.llm]['lm_head_device'])
    @property
    def device(self): return self.emb_device

    def to(self, device:torch.device):
        if self.model is not None:
            self.model = self.model.to(device)
        else: raise Exception('No model specified!')
    
    def __str__(self) -> str:
        """
        Print:
        model trainable parameters and 
        with number of trainable parameters
        """
        params = 'Parameters to be edited are Contained in these layers:\n'
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params = params + '    - ' + name + '     \t->   ' + str(param.size()) + '\n'
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params_count = sum([np.prod(p.size()) for p in model_parameters])
        params_count_M = (params_count/1024)/1024
        return params+'Trainable parameters count: {} -> {} MB\n'.format(params_count,params_count_M)


class Evaluator(object):
    @staticmethod
    def efficacy(model, batch_inputs:list, batch_prompts:list):
        if isinstance(batch_inputs, str): batch_inputs = [batch_inputs,]
        if isinstance(batch_prompts, str): batch_prompts = [batch_prompts,]
        inputs, num_seq, num_prompt, _ = model.tokenize(batch_inputs, batch_prompts)
        num_answer = list(map(lambda x,y : x-y, num_seq, num_prompt))
        with torch.no_grad():
            outputs = model(batch_inputs, batch_prompts)
        if type(outputs) is torch.Tensor: logits = outputs
        else: logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).tolist()
        answers = list(map(lambda x,i,j: x[i-1:i+j-1], answers,num_prompt,num_answer))
        labels = inputs['labels'].tolist()
        labels = list(map(lambda x,i,j: x[i:i+j], labels,num_prompt,num_answer))
        if -100 in labels: raise Exception('Error in labels when evaluation!')
        res = []
        for ans,label in zip(answers,labels):
            temp_acc = np.mean(np.equal(ans, label))
            if np.isnan(temp_acc): continue
            res.append(temp_acc)
        return res
    @staticmethod
    def locality(model, batch_inputs:list, batch_prompts:list, original_preds:list):
        if isinstance(batch_inputs, str): batch_inputs,batch_prompts = [batch_inputs,], [batch_prompts,]
        # Get the model predictions
        inputs, num_seq, num_prompt, _ = model.tokenize(batch_inputs, batch_prompts)
        num_answer = list(map(lambda x,y : x-y, num_seq, num_prompt))
        with torch.no_grad():
            outputs = model(batch_inputs, batch_prompts)
        if type(outputs) is torch.Tensor: logits = outputs
        else: logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).tolist()
        answers = list(map(lambda x,i,j: x[i-1:i+j-1], answers,num_prompt,num_answer))
        #
        # Obtain the original outputs
        labels = list(map(lambda x,i,j: x[i-1:i+j-1], original_preds,num_prompt,num_answer))
        res = []
        for ans,label in zip(answers,labels):
            temp_acc = np.mean(np.equal(ans, label))
            if np.isnan(temp_acc): continue
            res.append(temp_acc)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True, choices=['zsre', 'counterfact'])
    parser.add_argument('--reproduce', type=bool, default=False, help='Reproduce the original data to backup pre-generations.')
    parser.add_argument('--verbose', type=bool, default=False, help='Print the detailed evaluation results of each edit.')
    parser.add_argument('--epoch', type=int, default=15, help='Epoch number of the edit process.')
    parser.add_argument('--edit_bs', type=int, default=2, help='Batch size of edit.')
    parser.add_argument('--test_bs', type=int, default=2, help='Batch size of evaluation.')
    #
    parser.add_argument('--lamb', type=float, default=0.1, help='Lambda to balance contrastive loss.')
    parser.add_argument('--eta', type=int, default=2, help='Control the assessmention frequency.')
    parser.add_argument('--topk', type=int, default=15, help='Type the critical path number in PAC.')
    parser.add_argument('--bottomk', type=int, default=1, help='Type the non-critical path number in PAC.')
    args = parser.parse_args()
    #
    #
    model = PAC(args)
    # 4 GPU cards as configuration, you can configure it according to your requirements
    if model.model.config.num_hidden_layers == 28: # GPTJ
        gpu_config = {0:list(range(0,6)), 1:list(range(6,13)), 2:list(range(13,21)), 3:list(range(21, 28))}
    elif model.model.config.num_hidden_layers == 32: # Llama2 7B and Llama3 8B
        gpu_config = {0:list(range(0,8)), 1:list(range(8,16)), 2:list(range(16,24)), 3:list(range(24,32))}
    else: raise Exception('Model is not correct!')
    if torch.cuda.device_count == 1: model.to('cuda:0')
    else: model.model.disperse_to_gpus(gpu_config)
    #
    if args.data_name == 'zsre':
        data = ZsRE(CONFIG['data']['zsre'].format(''))
        data.segments(group_idx=0)
        if not os.path.exists(CONFIG['data']['zsre'].format('_'+model.llm)):
            data.reproduce(model.model, model.tokenizer, model.emb_device, './data/zsre_edit_'+model.llm+'.json')
        edit_loader = ZsREDataLoader(data, args.edit_bs)
        assess_loader = ZsREDataLoader(data, args.edit_bs)
        test_loader = ZsREDataLoader(data, args.test_bs, shuffle=False)
    elif args.data_name == 'counterfact':
        data = CounterFact(CONFIG['data']['counterfact'].format(''))
        data.segments(group_idx=0)
        if not os.path.exists(CONFIG['data']['counterfact'].format('_'+model.llm)):
            data.reproduce(model.model, model.tokenizer, model.emb_device, './data/counterfact_edit_'+model.llm+'.json')
        edit_loader = CounterFactDataLoader(data, args.edit_bs)
        assess_loader = CounterFactDataLoader(data, args.edit_bs)
        test_loader = CounterFactDataLoader(data, args.test_bs, shuffle=False)
    else: raise Exception(f"Cannot recognized the data name {args.data_name}!")
    #
    print(model)
    print(f'Model edit settings:\n', args, '\n', '--'*15)
    #
    #
    # Pre-Evaluation
    print('\nModel evaluation before the model editing.')
    efficacy_acc = list()
    rephrase_acc = list()
    locality_acc = list()
    pbar = tqdm(total=data.__len__(), ncols=75, leave=True)
    pbar.set_description_str(desc='Pre->')
    for _, batch in enumerate(test_loader):
        efficacy_acc.append(Evaluator.efficacy(model, batch[1], batch[2]))
        rephrase_acc.append(Evaluator.efficacy(model, batch[5], batch[6]))
        locality_acc.append(Evaluator.locality(model, batch[9], batch[10], batch[12]))
        pbar.update(len(batch[0]))
    pbar.refresh()
    print(f'\nMean Efficacy: {np.mean(flatten(efficacy_acc))}',
          f'\nMean Rephrase: {np.mean(flatten(rephrase_acc))}',
          f'\nMean Locality: {np.mean(flatten(locality_acc))}')
    #
    #
    # Model editing
    start_time = now()
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch+1}:')
        losses = list()
        if args.topk is not None: model.assessment(assess_loader, epoch)
        torch.cuda.empty_cache()
        pbar.set_description_str(desc='Edit->')
        pbar.reset()
        for _, batch in enumerate(edit_loader):
            loss = model.edit(batch[1], batch[2], batch[4])
            pbar.update(len(batch[0]))
            losses.append(loss)
            torch.cuda.empty_cache()
        pbar.refresh()
        print(f'\n[Mean Iter Loss ({epoch+1}/{args.epoch})]: {np.mean(flatten(losses))}')
        #
        #
        # Post-Evaluation
        case_ids = list()
        efficacy_acc = list()
        rephrase_acc = list()
        locality_acc = list()
        if not args.verbose:
            pbar.set_description_str(desc='Post->')
            pbar.reset()
        for idx, batch in enumerate(test_loader):
            case_ids.append(batch[1])
            efficacy_acc.append(Evaluator.efficacy(model, batch[1], batch[2]))
            rephrase_acc.append(Evaluator.efficacy(model, batch[5], batch[6]))
            locality_acc.append(Evaluator.locality(model, batch[9], batch[10], batch[12]))
            if not args.verbose: pbar.update(len(batch[0]))
            else: print(f'[{now()} ({idx+1}/{int(np.ceil(data.__len__()/args.test_bs))})]:',
                  f'\n\tEfficacy: {efficacy_acc[-1]}',
                  f'\n\tRephrase: {rephrase_acc[-1]}',
                  f'\n\tLocality: {locality_acc[-1]}')
        if not args.verbose: pbar.refresh()
        print(f'\n[Mean Efficacy ({epoch+1}/{args.epoch})]: {np.mean(flatten(efficacy_acc))}')
        print(f'[Mean Rephrase ({epoch+1}/{args.epoch})]: {np.mean(flatten(rephrase_acc))}')
        print(f'[Mean Locality ({epoch+1}/{args.epoch})]: {np.mean(flatten(locality_acc))}')
        end_time = now()
        print(f'[Epoch {epoch+1} Start/Finish  Time]: {start_time} / {end_time}')
    pbar.close()
    print(f'Model edit settings:\n', args, '\n', '--'*15+'\n')
