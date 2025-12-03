import argparse
import glob
import logging
import os

from config import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
import evaluation
import finetune_gpt2_additional_layer
import finetune_gpt2_vae

import random
import re
import shutil
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import generate
import Data2TextScorer as scorer
import copy
import json
from torch_scatter import scatter_min

if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_vae",
                "finetune_gpt2_additional_layer_with_prefix", "finetune_gpt2_vae_with_prefix"]:
    sts_model = SentenceTransformer(opt.sts_model)

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_constant_schedule,
    BatchEncoding
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def write_into_file(file_path, content):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    f = open(file_path, "a")
    f.write(content)
    f.close()


## from finetune_gpt2_additional_layer.py
def save_checkpoint(save_path, model):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict()}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))
        input_ids, labels, src, tgt, cate = zip(*examples)
        # print(len(input_ids), len(labels), len(weights))
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)
            if self.format_mode == 'cat':
                mode_input = 3
            elif self.format_mode == 'peek':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                mode_input = 2
            elif self.format_mode == 'infix':
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = (cate_batch != self.tokenizer.pad_token_id)

            labels[labels == self.tokenizer.pad_token_id] = -100  # tgt
            src_attn = (src != self.tokenizer.pad_token_id)  # src
            tgt_attn = (batch != self.tokenizer.pad_token_id)  # tgt

            if cate_batch is None:
                #     return [batch,labels,src_attn,tgt_attn,src]
                # return {"input_ids": batch, "labels": labels}
                # 'src_attn': src_attn, 'tgt_attn':tgt_attn,
            #             'src':src}
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn': tgt_attn, 'src': src}
            else:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn': tgt_attn,
                        'src': src, "cate_batch": cate_batch, "cate_attn": cate_attn}

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class LineByLineData2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, additional_special_tokens:list):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(additional_special_tokens[0], src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        if opt.rdf2sen_augment_data_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.evalaute_e2e_output_files_sts_roberta_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_e2e_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = str(row['bleu_pred'])
                str3 = row['input'].replace("<|endoftext|>","").strip()
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    src_lines.append(str1)

        elif opt.paraphrased_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.rdf2sen_aug_data_filter)
            elif opt.augment_tag == "prefix":
                # pd_reader = pd.read_csv(opt.evalaute_e2e_output_files_prefix_test_file_final)
                pd_reader = pd.read_csv(opt.evalaute_e2e_output_files_prefix_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_e2e_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = row['input'].replace("<|endoftext|>","").strip()
                str3 = str(row['bleu_pred']).replace("<|endoftext|>","").strip()
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    src_lines.append(str1)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1

        # print('tgt_avg: ', temp_tgt_len / temp_count)
        # print('src_avg: ', temp_src_len / temp_count)
        # print('ratios: ', temp_src_len/temp_tgt_len)
        #
        #
        #
        #
        # print(self.labels[0])
        # print(self.examples[0])
        # print(edited_sents[0])
        # print(self.src_sent[0])
        # print(self.tgt_sent[0])
        # print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )


class LineByLineTriplesTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, additional_special_tokens:list):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)


        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
            for src, tgt in zip(full_src_lst, full_tgt_lst):
                sent = ' {} {} {} '.format(additional_special_tokens[0], src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            for src, tgt in zip(full_src_lst, full_tgt_lst):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        if opt.rdf2sen_augment_data_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.evalaute_dart_output_files_sts_roberta_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_dart_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = str(row['bleu_pred'])
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = row['input'].replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)
        elif opt.webnlg_rdf2sen_augment_data_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_sts_roberta_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = str(row['bleu_pred'])
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = row['input'].replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)

        elif opt.paraphrased_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.rdf2sen_aug_data_filter)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = row['input'].replace("<|endoftext|>","").strip()
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = str(row['bleu_pred']).replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx

                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )


class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, additional_special_tokens:list):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)



        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                sent = ' {} {} {} '.format(additional_special_tokens[0], src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
            else:
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        if opt.rdf2sen_augment_data_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_sts_roberta_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = str(row['bleu_pred'])
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = row['input'].replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)
        elif opt.dart_rdf2sen_augment_data_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.evalaute_dart_output_files_sts_roberta_test_file)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_dart_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = str(row['bleu_pred'])
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = row['input'].replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)

        elif opt.paraphrased_tag == True:
            if opt.augment_tag == "sts-roberta":
                pd_reader = pd.read_csv(opt.rdf2sen_aug_data_filter)
            elif opt.augment_tag == "vae":
                pd_reader = pd.read_csv(opt.evalaute_dart_output_files_vae_test_file)
            for index, row in pd_reader.iterrows():
                str1 = row['input'].replace("<|endoftext|>","").strip()
                if str1[0] != '|':
                    str1 = ' | ' + str1
                if str1[0] == '|':
                    str1 = ' ' + str1

                str3 = str(row['bleu_pred']).replace("<|endoftext|>","").strip()
                rela_list2 = []
                if str1 != 'nan' and str1 != 'None' and str1 != '':
                    proc_sent = str1.split(" | ")
                    for proc in proc_sent:
                        proc_split = proc.split(" : ")
                        if len(proc_split) == 3:
                            rela_list2.append(proc_split[1])

                    if "prefix" in opt.flag and opt.rdf2sen_augment_data_tag == False and opt.sen2rdf_augment_data_tag == False:
                        sent = ' {} {} {} '.format(additional_special_tokens[0], str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)
                    else:
                        sent = ' {} {} '.format(str1, bos_tok) + str3 + ' {}'.format(eos_tok)
                        edited_sents.append(sent)

                    full_src_lst.append(str1)
                    full_tgt_lst.append(str3)
                    full_rela_lst.append(rela_list2)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        print(self.src_cat[1])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        if "e2e_data" in opt.train_data_file:
            return LineByLineData2TextTextDataset(tokenizer, file_path=file_path, block_size=args.block_size,
                                         bos_tok=tokenizer.bos_token, eos_tok=tokenizer.eos_token, additional_special_tokens=tokenizer.additional_special_tokens)
        if "DART" in opt.train_data_file:
            return LineByLineTriplesTextDataset(tokenizer, file_path=file_path, block_size=args.block_size,
                                         bos_tok=tokenizer.bos_token, eos_tok=tokenizer.eos_token, additional_special_tokens=tokenizer.additional_special_tokens)
        if "webnlg" in opt.train_data_file:
            return LineByLineWebNLGTextDataset(tokenizer, file_path=file_path, block_size=args.block_size,
                                         bos_tok=tokenizer.bos_token, eos_tok=tokenizer.eos_token, additional_special_tokens=tokenizer.additional_special_tokens)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


## from finetune_gpt2_additional_layer.py and finetune_gpt2_vae.py
def sts_stsb_roberta_large_representation(sent: [str], args):
    label_sts = sts_model.encode(sent, convert_to_tensor=True)
    latent_inputs = label_sts.reshape(len(label_sts), 12, 1, 64)
    size = len(latent_inputs.shape)
    re_size = [12, 2]
    for s in range(size):
        re_size.append(1)
    past_key_values = tuple(latent_inputs.repeat(re_size).to(args.device))

    return past_key_values


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    data_collator = DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=data_collator
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )

    scheduler = get_constant_schedule(optimizer)
    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    output_log_file = args.output_dir + '/loss_log.txt'

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
                # "input_ids": batch, "labels": labels, 'src_attn': src_attn,
                # 'tgt_attn': tgt_attn,
                #             'src':src
                batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
                '''inputs, labels, src_attn, tgt_attn,src= batch[0],batch[1],batch[2], batch[3],batch[4]
                inputs, labels, src_attn, tgt_attn,src = inputs.to(args.device), labels.to(args.device),
                    src_attn.to(args.device), tgt_attn.to(args.device), src.to(args.devics)'''
                batch = {x: batch[x] for x in batch if x in ('input_ids', 'labels')}
                model.train()
                if 't5' in opt.model_type:
                    batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
                    inputs = batch['input_ids']
                    labels = batch['labels']

                    ind = torch.where(torch.not_equal(labels, -100))
                    inputs_end = scatter_min(ind[1], ind[0])[0]

                    label_sents = []
                    input_data = []
                    for b_s in batch['labels']:
                        label_sents.append(
                            tokenizer.decode(b_s[b_s != -100]).strip())
                    for ind, i_s in zip(inputs_end, batch['input_ids']):
                        input_data.append(
                            tokenizer.decode(i_s[:ind]).strip())
                    input_ids = tokenizer.batch_encode_plus(input_data, padding=True,truncation=True,max_length=512,return_tensors='pt')['input_ids'].to(args.device)
                    labels = tokenizer.batch_encode_plus(label_sents,padding=True,truncation=True,max_length=512,return_tensors='pt')['input_ids'].to(args.device)
                    outputs = model(input_ids=input_ids, labels=labels)
                else:
                    outputs = model(**batch)
                # outputs = model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
                batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
                inputs = batch['input_ids']
                attention_mask = batch['tgt_attn']
                label_sents = []
                for b_s in batch['labels']:
                    label_sents.append(
                        model.tokenizer.decode(b_s[b_s != -100]).strip(model.tokenizer.eos_token).strip())
                sts_rep = sts_model.encode(label_sents, convert_to_tensor=True)
                sts_rep = sts_rep.to(args.device)

                model.train()
                loss = model(inputs, sts_rep, attention_mask)
            elif opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
                batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
                inputs = batch['input_ids']
                attention_mask = batch['tgt_attn']
                labels = batch['labels']

                ind = torch.where(torch.not_equal(labels, -100))
                inputs_end = scatter_min(ind[1], ind[0])[0]

                label_sents = []
                input_data = []
                for b_s in batch['labels']:
                    label_sents.append(
                        tokenizer.decode(b_s[b_s != -100]).strip(tokenizer.eos_token).strip())
                for ind, i_s in zip(inputs_end, batch['input_ids']):
                    input_data.append(
                        tokenizer.decode(i_s[:ind]).strip(tokenizer.eos_token).strip())

                model.train()
                loss = model(input_data=input_data, label_sents=label_sents, inputs=inputs, labels=labels, attention_mask=attention_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % len(train_dataloader) == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(global_step, args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / len(train_dataloader), global_step)

                    log_str = 'loss: ' + str((tr_loss - logging_loss) / len(train_dataloader)) \
                              + ' global_step: ' + str(global_step) + "\n"
                    write_into_file(output_log_file, log_str)

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % len(train_dataloader) == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
                        model_to_save.save_pretrained(output_dir)
                    elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_vae",
                                      "finetune_gpt2_additional_layer_with_prefix", "finetune_gpt2_vae_with_prefix"]:
                        save_checkpoint(output_dir + '/model.pt', model_to_save)
                        model.decoder.decoder.config.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(global_step, args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    data_collator = DataCollatorForData2TextLanguageModeling(tokenizer=tokenizer, mlm=False)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=data_collator
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    bleu_score = bleu_score_probing(model, tokenizer)
    # bleu_score = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
            batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
            '''inputs, labels, src_attn, tgt_attn,src= batch[0],batch[1],batch[2], batch[3],batch[4]
            inputs, labels, src_attn, tgt_attn,src = inputs.to(args.device), labels.to(args.device),
                src_attn.to(args.device), tgt_attn.to(args.device), src.to(args.devics)'''
            batch = {x: batch[x] for x in batch if x in ('input_ids', 'labels')}
            with torch.no_grad():
                if 't5' in opt.model_type:
                    batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
                    inputs = batch['input_ids']
                    labels = batch['labels']

                    ind = torch.where(torch.not_equal(labels, -100))
                    inputs_end = scatter_min(ind[1], ind[0])[0]

                    label_sents = []
                    input_data = []
                    for b_s in batch['labels']:
                        label_sents.append(
                            tokenizer.decode(b_s[b_s != -100]).strip())
                    for ind, i_s in zip(inputs_end, batch['input_ids']):
                        input_data.append(
                            tokenizer.decode(i_s[:ind]).strip())
                    input_ids = tokenizer.batch_encode_plus(input_data, padding=True,truncation=True,max_length=512,return_tensors='pt')['input_ids'].to(args.device)
                    labels = tokenizer.batch_encode_plus(label_sents,padding=True,truncation=True,max_length=512,return_tensors='pt')['input_ids'].to(args.device)
                    outputs = model(input_ids=input_ids, labels=labels)
                else:
                    outputs = model(**batch)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1
        elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
            batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
            inputs = batch['input_ids']
            attention_mask = batch['tgt_attn']
            label_sents = []
            for b_s in batch['labels']:
                label_sents.append(model.tokenizer.decode(b_s[b_s != -100]).strip(model.tokenizer.eos_token).strip())
            sts_rep = sts_model.encode(label_sents, convert_to_tensor=True)
            sts_rep = sts_rep.to(args.device)


            with torch.no_grad():
                lm_loss = model(inputs, sts_rep, attention_mask)
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1
        elif opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
            batch = {k: v.type(torch.long).to(args.device) for k, v in batch.items()}
            inputs = batch['input_ids']
            attention_mask = batch['tgt_attn']
            labels = batch['labels']

            ind = torch.where(torch.not_equal(labels, -100))
            inputs_end = scatter_min(ind[1], ind[0])[0]

            label_sents = []
            input_data = []
            for b_s in batch['labels']:
                label_sents.append(
                    tokenizer.decode(b_s[b_s != -100]).strip(tokenizer.eos_token).strip())
            for ind, i_s in zip(inputs_end, batch['input_ids']):
                input_data.append(
                    tokenizer.decode(i_s[:ind]).strip(tokenizer.eos_token).strip())

            with torch.no_grad():
                lm_loss = model(input_data=input_data, label_sents=label_sents, inputs=inputs, labels=labels,
                             attention_mask=attention_mask)
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_log_file = args.output_dir + '/loss_log.txt'
    log_str = 'eval_loss: ' + str(eval_loss) \
              + ' global_step: ' + str(global_step) + "\n"
    write_into_file(output_log_file, log_str)

    output_log_file = args.output_dir + '/bleu_scores.txt'
    log_str = 'Average BLEU score for the whole eval data: ' + str(bleu_score) \
              + ' global_step: ' + str(global_step) + "\n"
    write_into_file(output_log_file, log_str)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


# Created for testing BLEU Scores during the evaluate() step in finetune.py
def bleu_score_probing(model, tokenizer):
    logger.info("Prediction started")
    generation = generate.generate_predictions_for_bleu_score_probing(model, tokenizer)
    logger.info("Total size: %d", len(generation))
    logger.info("Revealing the first five...")
    logger.info("Prediction finished")
    logger.info("Evaluating these predictions by BLEU Score...")
    bleu_score = 0
    for item in generation:
        label = item[1]
        # skipping examples and labels
        predictions = item[2:]
        bleu_score += scorer.get_bleu_result(predictions, label)["bleu_score"]
    return round(bleu_score / len(generation), 3)


def finetune(paras):
    parser = argparse.ArgumentParser()
    print("The finetuned model will be saved at: ", opt.output_dir)
    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args(paras)

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    # rm after use ####################
    # sorted_checkpoints = _sorted_checkpoints(args)
    # for checkp in sorted_checkpoints:
    #     args.model_name_or_path = checkp
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,
    #                                               add_prefix_space=True)
    #     args.block_size = min(args.block_size, tokenizer.model_max_length)
    #     model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
    #     # Setup CUDA, GPU & distributed training
    #     if args.local_rank == -1 or args.no_cuda:
    #         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #         args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    #     else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #         torch.cuda.set_device(args.local_rank)
    #         device = torch.device("cuda", args.local_rank)
    #         torch.distributed.init_process_group(backend="nccl")
    #         args.n_gpu = 1
    #     args.device = device
    #     model.to(args.device)
    #
    #     if args.local_rank == 0:
    #         torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    #
    #     logger.info("Training/evaluation parameters %s", args)
    #
    #     # We don't add special tokens now.
    #
    #     special_tokens_dict = {'additional_special_tokens': ['<rdf2sen>']}
    #     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #     print("The number of add special tokens is: ", num_added_toks)
    #     if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix",
    #                     "finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
    #         model.decoder.decoder.resize_token_embeddings(len(tokenizer))
    #     else:
    #         model.resize_token_embeddings(len(tokenizer))
    #
    #     ##############################################################
    #     ################# ADJUST TOKENIZER ###########################
    #     ##############################################################
    #
    #     #   print(model_args.tuning_mode)
    #     print('adapting the size of the model embedding to include [PAD]')
    #     print('len(tokenizer) = ', len(tokenizer))
    #     num_added_tokens = tokenizer.add_special_tokens(
    #         {'pad_token': '[PAD]'})
    #     if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix",
    #                     "finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
    #         embedding_layer = model.decoder.decoder.resize_token_embeddings(len(tokenizer))
    #     else:
    #         embedding_layer = model.resize_token_embeddings(len(tokenizer))
    #     print('len(tokenizer) = ', len(tokenizer))
    #     print(tokenizer.eos_token, tokenizer.eos_token_id)
    #     print(tokenizer.bos_token, tokenizer.bos_token_id)
    #     global_step = checkp.rsplit('checkpoint-')[1]
    #     bleu_score = bleu_score_probing(model, tokenizer)
    #     output_log_file = args.output_dir + '/bleu_scores.txt'
    #     log_str = 'Average BLEU score for the whole eval data: ' + str(bleu_score) \
    #               + ' global_step: ' + str(global_step) + "\n"
    #     write_into_file(output_log_file, log_str)
    # return
    #######################################################
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir, add_prefix_space=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,
                                                  add_prefix_space=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.should_continue:
        if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
            model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
        elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
            model_gpt = AutoModelWithLMHead.from_pretrained(args.model_type)
            model_gpt.resize_token_embeddings(len(tokenizer))
            model = finetune_gpt2_additional_layer.Decoder(model_gpt, tokenizer)
            model = finetune_gpt2_additional_layer.load_checkpoint(load_path=args.model_name_or_path, model=model, device=device)
        elif opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
            model_gpt = AutoModelWithLMHead.from_pretrained(args.model_type)
            model_gpt.resize_token_embeddings(len(tokenizer))
            model = finetune_gpt2_vae.VariationalAutoencoder(device=device, model=model_gpt, tokenizer=tokenizer, latent_dims=32)
            model = finetune_gpt2_vae.load_checkpoint(load_path=args.model_name_or_path, model=model, device=device)
    elif args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>',
                               'additional_special_tokens': [' <rdf2sen> ']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
            model = finetune_gpt2_additional_layer.Decoder(model, tokenizer)
            model.to(args.device)
        elif opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
            model = finetune_gpt2_vae.VariationalAutoencoder(device=args.device, model=model, tokenizer=tokenizer, latent_dims=32)
            model.to(args.device)
    else:
        logger.info("Training new model from scratch")
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # We don't add special tokens now.

    special_tokens_dict = {'additional_special_tokens': ['<rdf2sen>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("The number of add special tokens is: ", num_added_toks)
    if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix",
                    "finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
        model.decoder.decoder.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    ##############################################################
    ################# ADJUST TOKENIZER ###########################
    ##############################################################

    #   print(model_args.tuning_mode)
    print('adapting the size of the model embedding to include [PAD]')
    print('len(tokenizer) = ', len(tokenizer))
    num_added_tokens = tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'})
    if opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix",
                    "finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
        embedding_layer = model.decoder.decoder.resize_token_embeddings(len(tokenizer))
    else:
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.bos_token, tokenizer.bos_token_id)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # train_dataset is a supposed to be a Dataset object.
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == '__main__':
    paras = ["--train_data_file", opt.train_data_file,
             "--eval_data_file", opt.eval_data_file,
             "--evaluate_during_training",
             "--model_type", opt.model_type,
             "--model_name_or_path", opt.model_type,
             "--do_train",
             "--do_eval",
             "--output_dir", opt.output_dir,
             "--num_train_epochs", opt.num_train_epochs,
             "--block_size", opt.block_size,
             "--save_steps", opt.save_steps,
             "--logging_steps", opt.logging_steps,
             "--learning_rate", opt.learning_rate,
             # "--gradient_accumulation_steps", '4',
             "--per_gpu_train_batch_size", opt.per_gpu_train_batch_size,
             "--per_gpu_eval_batch_size", opt.per_gpu_eval_batch_size,
             # "--should_continue",
             "--line_by_line"]
    # finetune(paras=paras)

    # generate.generate(model_type=opt.model_type,
    #                   model_path=opt.model_path,
    #                   test_data_path=opt.test_data_path,
    #                   logging_steps_generate=opt.logging_steps_generate,
    #                   output_path=opt.output_path)

    evaluation.batch_generate(
         input_path=opt.output_path,
         output_path=opt.evaluate_output_path,
         logging_steps=opt.logging_steps_evaluate
    )
    print("finished")
