import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead
from utils import hf_masked_encode, hf_reconstruction_prob_tok, fill_batch
from tqdm import tqdm
import json
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
import csv
from config import opt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def write_into_csv(file_path, content_list):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerows(content_list)

def gen_neighborhood(args):

    # initialize seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # load model and tokenizer
    r_model = AutoModelWithLMHead.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()

    # remove unused vocab and special ids from sampling
    softmax_mask = np.full(len(tokenizer.vocab), False)
    softmax_mask[tokenizer.all_special_ids] = True
    for k, v in tokenizer.vocab.items():
        if '[unused' in k:
            softmax_mask[v] = True

    # load the inputs and labels
    if 'e2e' in opt.train_data_file:
        print("The ssmba sampling e2e train file has been started.")
        with open(opt.train_data_file, encoding="utf-8") as f:
            full_src_tgt_list = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                            and len(line.split('||')) == 2)]
        output_path = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/e2e_data/generated_src1_train_paraphrased_ssmba_"+str(opt.num_return_sequences)+".csv"
        print("The paraphrased e2e train file has been finished.")

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
                    full_src_tgt_list.append([temp_triples, sent["lex"]])

        output_path = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017/generated_train_paraphrased_ssmba_" + str(opt.num_return_sequences) + ".csv"
        print("The sampling webnlg train file has been started.")

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
                full_src_tgt_list.append([temp_triples, sent['text']])

        output_path = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART/generated_dart_paraphrased_full_train_ssmba_" + str(opt.num_return_sequences) + ".csv"
        print("The ssmba sampling dart train file has been started.")

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
        sample = batch[0]
        sample_labels = batch[1]
        text = sample_labels

        generation = []
        for item in text:
            lines = [[item]]
            # lines = [tuple(item.strip().split('\t'))]
            # lines = [tuple(s.strip().split('\t')) for s in open(args.in_file).readlines()]
            num_lines = len(lines)
            lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]

            # load label file if it exists
            if args.label_file:
                labels = [s.strip() for s in open(args.label_file).readlines()]
                output_labels = True
            else:
                labels = [0] * num_lines
                output_labels = False

            # shard the input and labels
            if args.num_shards > 0:
                shard_start = (int(num_lines / args.num_shards) + 1) * args.shard
                shard_end = (int(num_lines / args.num_shards) + 1) * (args.shard + 1)
                lines = [s_list[shard_start:shard_end] for s_list in lines]
                labels = labels[shard_start:shard_end]

            final_outputs = []

            # sentences and labels to process
            sents = []
            l = []

            # number sentences generated
            num_gen = []

            # sentence index to noise from
            gen_index = []

            # number of tries generating a new sentence
            num_tries = []

            # next sentence index to draw from
            next_sent = 0

            sents, l, next_sent, num_gen, num_tries, gen_index = \
                fill_batch(args,
                           tokenizer,
                           sents,
                           l,
                           lines,
                           labels,
                           next_sent,
                           num_gen,
                           num_tries,
                           gen_index)

            # main augmentation loop
            while (sents != []):

                # remove any sentences that are done generating and dump to file
                for i in range(len(num_gen))[::-1]:
                    if num_gen[i] == args.num_samples or num_tries[i] > args.max_tries:

                        # get sent info
                        gen_sents = sents.pop(i)
                        num_gen.pop(i)
                        gen_index.pop(i)
                        label = l.pop(i)

                        # write generated sentences
                        for sg in gen_sents[1:]:
                            output_sent = '\t'.join([repr(val)[1:-1] for val in sg])
                            generation.append(output_sent)

                # fill batch
                sents, l, next_sent, num_gen, num_tries, gen_index = \
                    fill_batch(args,
                               tokenizer,
                               sents,
                               l,
                               lines,
                               labels,
                               next_sent,
                               num_gen,
                               num_tries,
                               gen_index)

                # break if done dumping
                if len(sents) == 0:
                    break

                # build batch
                toks = []
                masks = []

                for i in range(len(gen_index)):
                    s = sents[i][gen_index[i]]
                    tok, mask = hf_masked_encode(
                        tokenizer,
                        *s,
                        noise_prob=args.noise_prob,
                        random_token_prob=args.random_token_prob,
                        leave_unmasked_prob=args.leave_unmasked_prob,
                    )
                    toks.append(tok)
                    masks.append(mask)

                # pad up to max len input
                max_len = max([len(tok) for tok in toks])
                pad_tok = tokenizer.pad_token_id

                toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
                masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
                toks = torch.stack(toks)
                masks = torch.stack(masks)

                # load to GPU if available
                if torch.cuda.is_available():
                    toks = toks.cuda()
                    masks = masks.cuda()

                # predict reconstruction
                rec, rec_masks = hf_reconstruction_prob_tok(toks, masks, tokenizer, r_model, softmax_mask,
                                                            reconstruct=True, topk=args.topk)

                # decode reconstructions and append to lists
                for i in range(len(rec)):
                    rec_work = rec[i].cpu().tolist()
                    s_rec = [s.strip() for s in
                             tokenizer.decode([val for val in rec_work if val != tokenizer.pad_token_id][1:-1]).split(
                                 tokenizer.sep_token)]
                    s_rec = tuple(s_rec)

                    # check if identical reconstruction or empty
                    if s_rec not in sents[i] and '' not in s_rec:
                        sents[i].append(s_rec)
                        num_gen[i] += 1
                        num_tries[i] = 0
                        gen_index[i] = 0

                    # otherwise try next sentence
                    else:
                        num_tries[i] += 1
                        gen_index[i] += 1
                        if gen_index[i] == len(sents[i]):
                            gen_index[i] = 0

                # clean up tensors
                del toks
                del masks

        generation = [generation[i:i + opt.num_return_sequences] for i in
                      range(0, len(generation), opt.num_return_sequences)]

        for inp in range(len(sample_labels)):
            item = []
            item.append(sample[inp])
            item.append(sample_labels[inp])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shard', type=int, default=0,
            help='Shard of input to process. Output filename '
            'will have _${shard} appended.')

    parser.add_argument('--num-shards', type=int, default=1,
            help='Total number of shards to shard input file with.')

    parser.add_argument('--seed', type=int,
            help='Random seed to use for reconstruction and noising.')

    parser.add_argument('-m', '--model', type=str, default='bert-base-uncased',
            help='Name of HuggingFace BERT model to use for reconstruction,'
            ' or filepath to local model directory.')

    parser.add_argument('--tokenizer', type=str, default=None,
            help='Name of HuggingFace tokenizer to use for vocabulary'
            ' or filepath to local tokenizer. If None, uses the same'
            ' as model.')

    parser.add_argument('-i', '--in-file', type=str,
            help='Path of input text file for augmentation.'
            ' Inputs should be separated by newlines with tabs indicating'
            ' BERT <SEP> tokens.')

    parser.add_argument('-l', '--label-file',  type=str, default=None,
            help='Path of input label file for augmentation if using '
            ' label preservation.' )

    parser.add_argument('-o', '--output-prefix', type=str,
            help='Prefix path for output files, including augmentations and'
            ' preserved labels.')

    parser.add_argument('-p', '--noise-prob', type=float, default=0.15,
            help='Probability for selecting a token for noising.'
            ' Selected tokens are then masked, randomly replaced,'
            ' or left the same.')

    parser.add_argument('-r', '--random-token-prob', type=float, default=0.1,
            help='Probability of a selected token being replaced'
            ' randomly from the vocabulary.')

    parser.add_argument('-u', '--leave-unmasked-prob', type=float, default=0.1,
            help='Probability of a selected token being left'
            ' unmasked and unchanged.')

    parser.add_argument('-b', '--batch', type=int, default=8,
            help='Batch size of inputs to reconstruction model.')

    parser.add_argument('-n', '--num-samples', type=int, default=opt.num_return_sequences,
            help='Number of augmented samples to generate for each'
            ' input example.')

    parser.add_argument('-t', '--max-tries', type=int, default=10,
            help='Number of tries to generate a unique sample'
            ' before giving up.')

    parser.add_argument('--min-len', type=int, default=0,
            help='Minimum length input for augmentation.')

    parser.add_argument('--max-len', type=int, default=256,
            help='Maximum length input for augmentation.')

    parser.add_argument('--topk', '-k', type=int, default=opt.top_k,
            help='Top k to use for sampling reconstructed tokens from'
            ' the BERT model. -1 indicates unrestricted sampling.')

    args = parser.parse_args()

    if args.shard >= args.num_shards:
        raise Exception('Shard number {} is too large for the number'
            ' of shards {}'.format(args.shard, args.num_shards))

    if not args.tokenizer:
        args.tokenizer = args.model


    gen_neighborhood(args)
