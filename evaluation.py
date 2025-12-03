import Data2TextScorer as scorer
import pandas as pd
import os
import ast
from config import opt
from tqdm import tqdm

def batch_generate(input_path: str, output_path: str, logging_steps: int):
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
        score_result = scorer.batch_score(samples[i:i+logging_steps], labels[i:i+logging_steps], predictions[i:i+logging_steps])
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


if __name__ == '__main__':
    batch_generate(
        input_path=opt.output_path,
        output_path=opt.evaluate_output_path,
        logging_steps=opt.logging_steps_evaluate
    )
