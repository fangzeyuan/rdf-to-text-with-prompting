import csv
import pandas as pd
import os
from config import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
from sentence_transformers import SentenceTransformer
from finetune_gpt2_additional_layer import Decoder, load_checkpoint
from finetune_gpt2_vae import VariationalAutoencoder
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")


def write_into_csv(file_path, content_list):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerows(content_list)

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


"""def load_and_cache_examples(args, tokenizer, evaluate=False):

    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineData2TextTextDataset(tokenizer,  file_path=file_path, block_size=args.block_size,
                                     bos_tok=tokenizer.bos_token,eos_tok=tokenizer.eos_token)
    else:
        return None"""

def load_examples_and_label(test_data_path, tokenizer):
    """"
    src_text = []
    label_text = []
    data_samples = pd.read_csv(test_data_path,header = None).values.tolist()

    for sample in data_samples:
        # src_text.append(tokenizer.bos_token + sample[1] + tokenizer.additional_special_tokens[0])
        # src_text.append(sample[1] + ' RDF ')
        if opt.flag in [
                "finetune_gpt2_prefix", "finetune_gpt2_additional_layer_with_prefix", "finetune_gpt2_vae_with_prefix"]:
            src_text.append(sample[1] + ' RDF ')
        else:
            src_text.append(sample[1] + ' ')  # test without prefix
        label_text.append(sample[0])
    samples = [[samp, tgr_t] for samp, tgr_t in zip(src_text, label_text)]

    return samples"""



# Created for testing BLEU Scores during the evaluate() step in finetune.py
def generate_predictions_for_bleu_score_probing(model, tokenizer):
    test_datasets = load_examples_and_label(opt.eval_data_file, tokenizer).items()[:100]

    test_sampler = SequentialSampler(test_datasets)
    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=16
    )

    # model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader, desc="Generating predictions for the first 100 eval data"):
        # Have no idea why both batch[0] and batch[1] are tuples
        samples = [list(batch[0]), list(batch[1])]
        sample = samples[0]
        labels = samples[1]

        generation = get_model_generation(samples, model, tokenizer, opt.test_num_return_sequences)

        for inp in range(len(labels)):
            item = []
            item.append(sample[inp])
            item.append(labels[inp])
            item.extend(generation[inp])
            generation_list.append(item)
    return generation_list


def generate(model_type: str, model_path: str, test_data_path: str, output_path: str):
    print("The generated result saved path is: ", output_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
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
    test_dataset_dict= read_e2e_files(test_data_path, tokenizer)
    #test_datasets = load_and_cache_examples(test_data_path, tokenizer)
    test_dataset =test_dataset_dict.items()

    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            csv_read = csv.reader(f)
            steps_generated = len(list(csv_read)) - 1
            test_datasets = test_datasets[steps_generated:]
    else:
        columns_list = ["sample", "label"]
        for i in range(0, opt.num_return_sequences):
            columns_list.append("generation_" + str(i))
        write_into_csv(output_path, [columns_list])

    def collate_tokenize(samples):
        sample = samples[0]
        tokenized = tokenizer(sample, padding='True', return_tensors='pt')
        return tokenized
    test_sampler = SequentialSampler(test_datasets)
    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=opt.test_batch_size,collate_fn=collate_tokenize
    )
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        samples = batch

        sample = samples[0]
        labels = samples[1]
        generation = get_model_generation(samples, model, tokenizer)

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
    print("\nGeneration Finished!")

def get_model_generation(samples, model, tokenizer, num_return_sequences=opt.num_return_sequences):
    sample = samples[0]

    def collate(sample):
        examples = sample[0]
        labels = sample[1]
        data_set = tokenizer.batch_encode_plus(examples, return_tensors="pt", padding=True)
        samples = data_set["input_ids"]
        attention_mask = data_set["attention_mask"]
        sts_model = SentenceTransformer(opt.sts_model)
        sts_rep = sts_model.encode(labels, convert_to_tensor=True)
        return [samples, sts_rep, attention_mask]

    if opt.flag in ["finetune_gpt2_vae", "finetune_gpt2_vae_with_prefix"]:
        generation_output = model.generate(inputs=sample, num_return_sequences=num_return_sequences)
    elif opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
        inputs = tokenizer.batch_encode_plus(sample, return_tensors="pt", padding=True)
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
        [inputs, sts_rep, attention_mask] = collate(samples)
        generation_output = model.generate(inputs=inputs.to(device),
                                           min_length=opt.min_length,
                                           max_length=opt.max_length,
                                           do_sample=opt.do_sample,
                                           past_key_values=sts_rep.to(device),
                                           attention_mask=attention_mask.to(device),
                                           num_return_sequences=num_return_sequences)

    generation = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    generation = [generation[i:i + num_return_sequences] for i in range(0, len(generation), num_return_sequences)]
    return [[m_i.replace(n.replace(tokenizer.bos_token, '').replace(tokenizer.additional_special_tokens[0], ''),
                               "").strip() for m_i in m] for n, m in zip(sample, generation)]

