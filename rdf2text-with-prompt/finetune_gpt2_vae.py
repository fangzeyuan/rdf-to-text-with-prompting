import argparse
import glob
import logging

from config import opt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

sts_model = SentenceTransformer(opt.sts_model)
# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0

from transformers import (
    BertModel,
    BertTokenizer,
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Encoder(nn.Module):
    def __init__(self, device, encoder_embeddings=768, latent_dims=768):
        super(Encoder, self).__init__()
        self.encoder_bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sts_linear = nn.Linear(in_features=encoder_embeddings, out_features=encoder_embeddings)
        self.fc_mu = nn.Linear(in_features=encoder_embeddings, out_features=latent_dims)
        self.fc_var = nn.Linear(in_features=encoder_embeddings, out_features=latent_dims)
        self.device = device

    def forward(self, inputs, label):
        inputs = [inp.replace(' RDF ', '') for inp in inputs]
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(self.device)
        out = self.encoder_bert(inputs, return_dict=True, output_hidden_states=True)
        out = out.hidden_states[-1][:, 0, :]
        input_sts_rep = self.sts_linear(out)
        out_mu = self.fc_mu(input_sts_rep)
        out_var = self.fc_var(input_sts_rep)
        kl_loss = self.kl_loss(out_mu, out_var)

        sts_loss = self.sts_loss(input_sts_rep, label)
        return out_mu, out_var, kl_loss+sts_loss

    def kl_loss(self, latent_mu, latent_var):
        kl_div = -0.5 * torch.sum(1 + latent_var - latent_mu.pow(2) - latent_var.exp())

        return kl_div

    def sts_loss(self, sts_rep, label, variational_beta=1):
        label = sts_model.encode(label, convert_to_tensor=True).to(self.device)
        loss_f = nn.MSELoss()
        loss = loss_f(sts_rep, label)

        return loss

    def generate(self, inputs):
        inputs = [inp.replace(' RDF ', '') for inp in inputs]
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True)
        inputs = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        out = self.encoder_bert(inputs, return_dict=True, output_hidden_states=True)
        out = out.hidden_states[-1][:, 0, :]
        input_sts_rep = self.sts_linear(out)
        out_mu = self.fc_mu(input_sts_rep)
        out_var = self.fc_var(input_sts_rep)

        return out_mu, out_var


class Decoder(nn.Module):
    def __init__(self, model, tokenizer, decoder_embeddings=768, latent_dims=768):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(in_features=latent_dims, out_features=decoder_embeddings)
        self.decoder = model
        self.tokenizer = tokenizer
        # self.drop = nn.Dropout(0.1)

    def forward(self, text, sts_rep, attention_mask):
        inputs = self.fc(sts_rep)
        inputs = inputs.reshape(len(inputs), 12, 1, 64)
        size = len(inputs.shape)
        re_size = [12, 2]
        for s in range(size):
            re_size.append(1)
        past_key_values = tuple(inputs.repeat(re_size))
        out = self.decoder(text, labels=text, past_key_values=past_key_values,
                           output_hidden_states=True, return_dict=True)
        return out.loss

    def generate(self, inputs, past_key_values, num_return_sequences = opt.num_return_sequences):
        inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        latent_inputs = self.fc(past_key_values)
        latent_inputs = latent_inputs.reshape(len(latent_inputs), 12, 1, 64)
        size = len(latent_inputs.shape)
        re_size = [12, 2]
        for s in range(size):
            re_size.append(1)
        past_key_values = tuple(latent_inputs.repeat(re_size))

        outputs = self.decoder.generate(inputs["input_ids"].to(self.decoder.device),
                                        min_length=opt.min_length,
                                        max_length=opt.max_length,
                                        do_sample=opt.do_sample,
                                        top_p=opt.top_p, top_k=opt.top_k,
                                        past_key_values=past_key_values,
                                        attention_mask=inputs["attention_mask"].to(self.decoder.device),
                                        num_return_sequences=num_return_sequences)
        return outputs


class VariationalAutoencoder(nn.Module):
    def __init__(self, device, model, tokenizer,
                 encoder_embeddings=768, latent_dims=256, decoder_embeddings=768, variational_beta=0.5):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(device, encoder_embeddings, latent_dims)
        self.decoder = Decoder(model, tokenizer, decoder_embeddings, latent_dims)
        self.variational_beta = variational_beta

    def forward(self, inputs, labels, samples, attention_mask):
        latent_mu, latent_var, loss = self.encoder(inputs, labels)
        latent = self.latent_sample(latent_mu, latent_var)
        reconstruct_loss = self.decoder(sts_rep=latent, text=samples, attention_mask=attention_mask)
        print("loss: ", loss, "  reconstruct_loss: ", reconstruct_loss)
        return self.variational_beta*loss + (1-self.variational_beta)*reconstruct_loss

    def generate(self, inputs, num_return_sequences=opt.num_return_sequences):
        latent_mu, latent_var = self.encoder.generate(inputs)
        latent = self.latent_sample(latent_mu, latent_var)
        generation = self.decoder.generate(past_key_values=latent, inputs=inputs, num_return_sequences=num_return_sequences)
        return generation

    def latent_sample(self, mu, var):
        if self.training:
            # the reparameterization trick
            std = var.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def save_checkpoint(save_path, model):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict()}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path + '/model.pt', map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])

    return model


def write_into_file(file_path, content):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    f = open(file_path, "a")
    f.write(content)
    f.close()


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        src_text = []
        tgr_text = []
        rdf_text = []
        data_samples = pd.read_csv(file_path).values.tolist()
        for sample in data_samples:
            rdf_text.append(sample[1])
            # src_text.append(sample[1] + ' RDF ' + sample[0])
            src_text.append(sample[1] + ' ' + sample[0]) # check whether can generate text without prefix
            tgr_text.append(sample[0])
        samples = [[rdf, samp, tgr_t] for rdf, samp, tgr_t in zip(rdf_text, src_text, tgr_text)]

        return samples


def sts_stsb_roberta_large_representation(sent: [str], args):
    label_sts = sts_model.encode(sent, convert_to_tensor=True)
    latent_inputs = label_sts.reshape(len(label_sts), 12, 1, 64)
    size = len(latent_inputs.shape)
    re_size = [12, 2]
    for s in range(size):
        re_size.append(1)
    past_key_values = tuple(latent_inputs.repeat(re_size).to(args.device))

    return past_key_values


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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

    vae_model = VariationalAutoencoder(device=args.device, model=model, tokenizer=tokenizer, latent_dims=32)
    vae_model.to(args.device)

    def collate(sample):
        rdfs = [item[0] for item in sample]
        examples = [item[1] for item in sample]
        labels = [item[2] for item in sample]
        data_set = tokenizer.batch_encode_plus(examples, return_tensors="pt", padding=True)
        samples = data_set["input_ids"]
        sam_attention_mask = data_set["attention_mask"]
        return [rdfs, labels, samples, sam_attention_mask]

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    vae_model = vae_model.module if hasattr(vae_model,
                                            "module") else vae_model  # Take care of distributed/parallel training
    vae_model.decoder.decoder.resize_token_embeddings(len(vae_model.decoder.tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in vae_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in vae_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
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

    vae_model.zero_grad()
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

            rdfs, labels, samples, attention_mask = batch[0], batch[1], batch[2], batch[3]
            samples = samples.to(args.device)
            attention_mask = attention_mask.to(args.device)
            vae_model.train()
            loss = vae_model(inputs=rdfs, labels=labels, samples=samples, attention_mask=attention_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                vae_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(global_step, args, vae_model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    log_str = 'loss: ' + str((tr_loss - logging_loss) / args.logging_steps) \
                              + ' global_step: ' + str(global_step) + "\n"
                    write_into_file(output_log_file, log_str)

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        vae_model.module if hasattr(vae_model, "module") else vae_model
                    )  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)
                    save_checkpoint(output_dir + '/model.pt', model_to_save)
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


def evaluate(global_step, args, model, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    def collate(sample):
        rdfs = [item[0] for item in sample]
        examples = [item[1] for item in sample]
        labels = [item[2] for item in sample]
        data_set = tokenizer.batch_encode_plus(examples, return_tensors="pt", padding=True)
        samples = data_set["input_ids"]
        sam_attention_mask = data_set["attention_mask"]
        return [rdfs, labels, samples, sam_attention_mask]

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
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

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        rdfs, labels, samples, attention_mask = batch[0], batch[1], batch[2], batch[3]
        samples = samples.to(args.device)
        attention_mask = attention_mask.to(args.device)

        with torch.no_grad():
            lm_loss = model(inputs=rdfs, labels=labels, samples=samples, attention_mask=attention_mask)
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_log_file = args.output_dir + '/loss_log.txt'
    log_str = 'eval_loss: ' + str(eval_loss) \
              + ' global_step: ' + str(global_step) + "\n"
    write_into_file(output_log_file, log_str)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def finetune(paras):
    parser = argparse.ArgumentParser()

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
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
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

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>',
                           'additional_special_tokens': [' <rdf2sen> ']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #         )
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    #
    #         model = AutoModelWithLMHead.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(global_step, args, model, tokenizer, prefix=prefix)
    #         result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    #         results.update(result)

    return results




if __name__ == '__main__':
    paras = ["--train_data_file", opt.train_data_file,
             "--eval_data_file", opt.eval_data_file,
             "--evaluate_during_training",
             "--model_type", opt.model_type,
             "--model_name_or_path", opt.model_type,
             "--do_train",
             "--do_eval",
             "--output_dir", opt.output_dir_vae,
             "--num_train_epochs", opt.num_train_epochs,
             "--block_size", opt.block_size,
             "--save_steps", opt.save_steps,
             "--logging_steps", opt.logging_steps,
             "--learning_rate", opt.learning_rate,
             # "--gradient_accumulation_steps", opt.gradient_accumulation_steps,
             "--per_gpu_train_batch_size", opt.per_gpu_train_batch_size,
             "--per_gpu_eval_batch_size", opt.per_gpu_eval_batch_size]

    # finetune(paras=paras)

    # generate.generate(model_type=opt.model_type,
    #          model_path=opt.model_path_vae,
    #          test_data_path=opt.test_data_path,
    #          batch_size=opt.batch_size,
    #          output_path=opt.output_path_vae)
    print("finished")
