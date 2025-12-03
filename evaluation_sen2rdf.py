from pathlib import Path

import csv

import Data2TextScorer as scorer
import pandas as pd
import os
import ast
from config import opt
from tqdm import tqdm

def batch_generate(input_path: str, output_path: str, logging_steps: int, text2data=False):
    print("The evaluation result saved path is: ", output_path)
    samples, labels = load_examples_and_label(input_path)
    predictions = load_predictions(input_path)
    begin_tag = 0
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if df.shape[0] > 0:
            begin_tag = df.shape[0]
    for i in tqdm(range(begin_tag, len(samples), logging_steps)):
        results = []
        score_result = scorer.batch_score(samples[i:i+logging_steps], labels[i:i+logging_steps], predictions[i:i+logging_steps], text2data=text2data)
        results.extend(score_result)
        write_into_csv(output_path, results)


def load_examples_and_label(test_data_path):
    src_text = []
    label_text = []
    data_samples = pd.read_csv(test_data_path).values.tolist()
    for sample in data_samples:
        src_text.append(sample[0])
        labels = []
        sample_list = ast.literal_eval(sample[1])
        for label in sample_list:
            labels.append(''.join(label))
        label_text.append(labels)

    return src_text, label_text

def load_predictions(pred_path):
    data = pd.read_csv(pred_path).values.tolist()
    predictions = []
    for row in data:
        # Skip input and label columns
        prediction = row[2:]
        prediction_list = []
        prediction_list_2 = []
        for i in prediction:
            prediction_list.append(str(i))
        for i in prediction_list:
            if i == 'nan':
                prediction_list_2.append("")
            else:
                prediction_list_2.append(i)
        predictions.append(prediction_list_2)
    return predictions


def write_into_csv(output_path, results):
    df = pd.DataFrame(results)
    append = False
    if os.path.isfile(output_path):
        append = True
    df.to_csv(output_path, index=False, mode='a' if append else 'w', header=not append)


# if __name__ == '__main__':
#     if "DART" in opt.train_data_file and opt.augment_tag == 'sts-roberta':
#         input_path = opt.dart_output_files_sts_roberta_test_file
#         output_path = opt.evalaute_dart_output_files_sts_roberta_test_file
#     elif "DART" in opt.train_data_file and opt.augment_tag == 'vae':
#         input_path = opt.dart_output_files_vae_test_file
#         output_path = opt.evalaute_dart_output_files_vae_test_file
#     elif "webnlg" in opt.train_data_file and opt.augment_tag == 'sts-roberta':
#         input_path = opt.webnlg_output_files_sts_roberta_test_file
#         output_path = opt.evalaute_webnlg_output_files_sts_roberta_test_file
#     elif "webnlg" in opt.train_data_file and opt.augment_tag == 'vae':
#         input_path = opt.webnlg_output_files_vae_test_file
#         output_path = opt.evalaute_webnlg_output_files_vae_test_file
#     elif "e2e" in opt.train_data_file and opt.augment_tag == 'sts-roberta':
#         input_path = opt.e2e_output_files_sts_roberta_test_file
#         output_path = opt.evalaute_e2e_output_files_sts_roberta_test_file
#     elif "e2e" in opt.train_data_file and opt.augment_tag == 'vae':
#         input_path = opt.e2e_output_files_vae_test_file
#         output_path = opt.evalaute_e2e_output_files_vae_test_file
#
#     batch_generate(
#         input_path=input_path,  #output_path
#         output_path=output_path,  #evaluate_output_path
#         logging_steps=opt.logging_steps_evaluate
#     )


def clean_generation(input_path: str, output_path: str, logging_steps=1):
    print("The cleanup result saved path is: ", output_path)
    bos_token = '<|endoftext|>'
    samples, labels = load_examples_and_label(input_path)
    predictions = load_predictions(input_path)

    content_list = [['sample', 'label', 'generation_0', 'generation_1','generation_2', 'generation_3', 'generation_4']]
    Path(output_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        writer = csv.writer(f, delimiter =",")
        writer.writerows(content_list)


    for i in tqdm(range(len(samples))):
        results = []
        gen_re = []

        int_to_rmv = samples[i].replace(bos_token, '').replace(' .', '.').replace(' ,', ',').replace(' \'', '\'').strip()
        for m_i in predictions[i]:
            gen_re.append(m_i.replace(int_to_rmv, '').strip())
        results.append(samples[i])
        results.append(labels[i])
        results.extend(gen_re)
        write_into_csv(output_path, [results])


if __name__ == '__main__':
    batch_generate(
        input_path='data/e2e_data/model_gpt2_fc_layer_without_prefix/checkpoint-8414/train_result.csv',
        # output_path
        output_path='data/e2e_data/model_gpt2_fc_layer_without_prefix/checkpoint-8414/evaluation_train_result.csv',
        # evaluate_output_path
        logging_steps=opt.logging_steps_evaluate,
        text2data=True
        )
