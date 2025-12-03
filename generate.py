import csv
import os

from config import opt

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
from sentence_transformers import SentenceTransformer
from finetune_gpt2_additional_layer import Decoder, load_checkpoint
from finetune_gpt2_vae import VariationalAutoencoder
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelWithLMHead, AutoTokenizer
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SequentialSampler,Dataset
from pathlib import Path
import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")


def write_into_csv(file_path, content_list):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        writer = csv.writer(f, delimiter =",")
        writer.writerows(content_list)


## for e2e dataset, there might be several labels for each rdf.
def read_e2e_files(path, tokenizer, lowdata_token=None):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('||')
            # URGENT CHANGE
            # src =  src + ' {}'.format(' summarize :')
            if lowdata_token is None:
                src = ' {} {}'.format(src, tokenizer.bos_token)
                # src =  src + ' {}'.format(tokenizer.bos_token)
            else:
                src = ' {} {} {}'.format(lowdata_token, src, tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict

def read_DART_files(path, tokenizer):
    file_dict = {}
    with open(path) as f:
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
            if sent['text'] != "":
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)

    for src, tgt in zip(full_src_lst, full_tgt_lst):
        sent = ' {} {}'.format(src, tokenizer.bos_token)
        if sent not in file_dict:
            file_dict[sent] = []
        file_dict[sent].append(tgt)
    return file_dict


def read_webnlg_files(path, tokenizer):
    file_dict = {}
    with open(path) as f:
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

    for src, tgt in zip(full_src_lst, full_tgt_lst):
        sent = ' {} {}'.format(src, tokenizer.bos_token)
        if sent not in file_dict:
            file_dict[sent] = []
        file_dict[sent].append(tgt)

    return file_dict


# The TestDataset is design for dataloader.
class TestDataset(Dataset):

    def __init__(self, dict_dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.src = list(dict_dataset.keys())
        self.trg = list(dict_dataset.values())

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        src = self.src[idx]
        trg = self.trg[idx]

        return src, trg


# Created for testing BLEU Scores during the evaluate() step in finetune.py

def generate_predictions_for_bleu_score_probing(model, tokenizer):
    # test_datasets = load_examples_and_label(opt.eval_data_file, tokenizer).items()[:100]
    if "e2e" in opt.train_data_file:
        test_dataset_dict = read_e2e_files(opt.eval_data_file, tokenizer)
    elif "DART" in opt.train_data_file:
        test_dataset_dict = read_DART_files(opt.eval_data_file, tokenizer)
    elif "webnlg" in opt.train_data_file:
        test_dataset_dict = read_webnlg_files(opt.eval_data_file, tokenizer)
    # test_dataset_dict_100 = {k: test_dataset_dict[k] for k in sorted(test_dataset_dict.keys())[:100]}
    # test_datasets = TestDataset(test_dataset_dict_100)
    test_datasets = TestDataset(test_dataset_dict)

    test_sampler = SequentialSampler(test_datasets)
    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=1
    )

    # model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader.dataset, desc="Generating predictions for the whole eval data"):
        # Have no idea why both batch[0] and batch[1] are tuples
        samples = [batch[0], batch[1]]
        sample = samples[0]
        labels = samples[1]

        generation = get_model_generation(sample, labels, model, tokenizer, opt.test_num_return_sequences)

        item = []
        item.append(batch[0])
        item.append(batch[1])
        for inp in generation:
            item.extend(inp)
        generation_list.append(item)
    return generation_list


# We set batch size = 1 for the generation to make sure we do not need to consider the alignment of each tensor in batch.
def generate(model_type: str, model_path: str, test_data_path: str, logging_steps_generate: str, output_path: str):
    print("The generated result saved path is: ", output_path)
    if 't5' in opt.model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
        if 't5' in opt.model_type:
            model = AutoModelWithLMHead.from_pretrained(model_path)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_path)
    elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
        model_gpt = GPT2LMHeadModel.from_pretrained(model_type)
        model_gpt.resize_token_embeddings(len(tokenizer))
        model = Decoder(model_gpt, tokenizer)
        model = load_checkpoint(load_path=model_path, model=model, device=device)
    elif opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
        model_gpt = GPT2LMHeadModel.from_pretrained(model_type)
        model_gpt.resize_token_embeddings(len(tokenizer))
        model = VariationalAutoencoder(device=device, model=model_gpt, tokenizer=tokenizer, latent_dims=32)
        model = load_checkpoint(load_path=model_path, model=model, device=device)
    if "e2e" in opt.train_data_file:
        test_dataset_dict = read_e2e_files(test_data_path, tokenizer)
    elif "DART" in opt.train_data_file:
        test_dataset_dict = read_DART_files(test_data_path, tokenizer)
    elif "webnlg" in opt.train_data_file:
        test_dataset_dict = read_webnlg_files(test_data_path, tokenizer)

    test_datasets = TestDataset(test_dataset_dict)

    columns_list = ["sample", "label"]
    for i in range(0, opt.num_return_sequences):
        columns_list.append("generation_" + str(i))
    write_into_csv(output_path, [columns_list])


    test_sampler = SequentialSampler(test_datasets)

    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=1 # ,collate_fn=collate_tokenize
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader.dataset, desc="Testing"):

        generation = get_model_generation(batch[0], batch[1], model, tokenizer)

        item = []
        item.append(batch[0])
        item.append(batch[1])
        for inp in generation:
            item.extend(inp)
        generation_list.append(item)
        if len(generation_list) >= opt.logging_steps_generate:
            write_into_csv(output_path, generation_list)
            generation_list = []

    write_into_csv(output_path, generation_list)
    print("\nGeneration Finished!")

# This function will generate the number of num_returen_sequences sentences.
def get_model_generation(sample, label, model, tokenizer, num_return_sequences=opt.num_return_sequences):

    def collate(sample):
        examples = sample
        labels = label
        data_set = tokenizer.batch_encode_plus([examples], return_tensors="pt", padding=True)
        samples = data_set["input_ids"]
        attention_mask = data_set["attention_mask"]
        sts_model = SentenceTransformer(opt.sts_model)
        sts_rep = sts_model.encode(labels, convert_to_tensor=True)
        return [samples, sts_rep, attention_mask]

    if opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
        generation_output = model.generate(inputs=[sample], num_return_sequences=num_return_sequences)
    elif opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":

        # inputs = tokenizer.batch_encode_plus(sample,  return_tensors="pt", padding=True)
        inputs = tokenizer(sample, return_tensors="pt", padding=True)
        # print(inputs)
        generation_output = model.generate(inputs["input_ids"].to(device),
                                           min_length=opt.min_length,
                                           max_length=opt.max_length,
                                           do_sample=opt.do_sample,
                                           top_p=opt.top_p, top_k=opt.top_k,
                                           bos_token_id=tokenizer.bos_token_id,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.pad_token_id,
                                           num_return_sequences=num_return_sequences)

    elif opt.flag in ["finetune_gpt2_additional_layer", "finetune_gpt2_additional_layer_with_prefix"]:
        [inputs, sts_rep, attention_mask] = collate(sample)
        generation_output = model.generate(inputs=inputs.to(device),
                                           min_length=opt.min_length,
                                           max_length=opt.max_length,
                                           do_sample=opt.do_sample,
                                           past_key_values=sts_rep.to(device),
                                           attention_mask=attention_mask.to(device),
                                           num_return_sequences=num_return_sequences)


    generation = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    generation = [generation[i:i + num_return_sequences] for i in range(0, len(generation), num_return_sequences)]

    if type(sample) == str:
        sample = [sample]
    return [[m_i.replace(n.replace(tokenizer.bos_token, ''),
                         "").strip() for m_i in m] for n, m in zip(sample, generation)]

