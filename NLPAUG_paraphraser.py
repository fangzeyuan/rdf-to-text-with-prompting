import os
from config import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
import csv
np.random.seed(42)
import random
random.seed(42)

from tqdm import tqdm
import json
import nlpaug.augmenter.word as naw

def NLPAUG_noising_function(text):
    flag = random.randint(1, 4)
    if flag == 1:
        aug = naw.RandomWordAug(action="delete")
    elif flag == 2:
        aug = naw.RandomWordAug(action="swap")
    elif flag == 3:
        aug = naw.WordEmbsAug(action="insert", model_type="glove", model_path='./glove.6B.50d.txt')
    elif flag == 4:
        aug = naw.RandomWordAug(action="substitute")
    augmented_text = aug.augment(text, n=opt.num_return_sequences)
    return augmented_text

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def write_into_csv(file_path, content_list):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerows(content_list)

if 'e2e' in opt.train_data_file:
    with open(opt.train_data_file, encoding="utf-8") as f:
        full_src_tgt_list = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                        and len(line.split('||')) == 2)]
    output_path = "data/e2e_data/generated_src1_train_paraphrased_NLPAUG_"+str(opt.num_return_sequences)+"_updated.csv"
    print("The paraphrased e2e train file has been started.")

elif 'dart' in opt.train_data_file:
    with open(opt.train_data_file) as f:
        lines_dict = json.load(f)

    full_src_tgt_list = []
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
            full_src_tgt_list.append([temp_triples,sent['text']])

    output_path = "data/DART/generated_dart_paraphrased_full_train_NLPAUG_"+str(opt.num_return_sequences)+"_updated.csv"
    print("The paraphrased DART train file has been started.")

elif 'webnlg' in opt.train_data_file:
    with open(opt.train_data_file) as f:
        lines_dict = json.load(f)

    full_src_tgt_list = []
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
                full_src_tgt_list.append([temp_triples,sent["lex"]])

    output_path = "data/webnlg_challenge_2017/generated_train_paraphrased_NLPAUG_" + str(opt.num_return_sequences) + "_updated.csv"
    print("The paraphrased webnlg train file has been started.")

if os.path.isfile(output_path):
    with open(output_path, "r") as f:
        csv_read = csv.reader(f)
        steps_generated = len(list(csv_read)) - 1
        full_src_tgt_list = full_src_tgt_list[steps_generated:]
else:
    columns_list = ["sample", "label"]
    for i in range(0, opt.num_return_sequences):
        columns_list.append("generation_" + str(i))
    write_into_csv(output_path, [columns_list])

test_sampler = SequentialSampler(full_src_tgt_list)
test_dataloader = DataLoader(
    full_src_tgt_list, sampler=test_sampler, batch_size=opt.test_batch_size
)
generation_list = []
for batch in tqdm(test_dataloader, desc="Testing"):
    samples = batch

    sample = samples[0]
    labels = samples[1]
    text = labels
    max_len = opt.max_length

    generation = []
    for item in text:
        temp_generation = NLPAUG_noising_function(item)
        generation.extend(temp_generation)

    generation = [generation[i:i + opt.num_return_sequences] for i in
                  range(0, len(generation), opt.num_return_sequences)]

    for inp in range(len(labels)):
        item = []
        item.append(sample[inp])
        item.append(labels[inp])
        item.extend(generation[inp])
        generation_list.append(item)
    if len(generation_list) >= opt.logging_steps_generate:
        write_into_csv(output_path, generation_list)
        generation_list = []
write_into_csv(output_path, generation_list)
if 'dart' in opt.train_data_file:
    print("\nThe paraphrased DART train file has been finished.")
elif 'webnlg' in opt.train_data_file:
    print("\nThe paraphrased webnlg train file has been finished.")
elif 'e2e' in opt.train_data_file:
    print("\nThe paraphrased e2e train file has been finished.")