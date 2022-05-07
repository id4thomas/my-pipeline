from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import csv
import pickle
import argparse
import sys

import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

# Utilities
from utils.data_utils import CLFDataset
from utils.perf_utils import *
from utils.utils import Config
from tqdm import tqdm, trange

def main(args):    
    #### Load Config
    config_dir = Path(args.config_dir) 
    config = Config(json_path=config_dir /  'config.json')


    #### Load Test Data
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    
    file_name = args.data.split('/')[-1].split('.')[0]

    if not os.path.exists(args.data):
        raise AssertionError(f"cannot find the file '{args.data}'")

    cwd = Path.cwd()
    test_fpath = cwd / args.data
    # test_filename = Path(args.data).stem

    if not test_fpath.exists():
        raise AssertionError("cannot find the file '%s'" % test_fpath)

    df_test = pd.read_csv(str(test_fpath), sep="\t")

    sents = df_test["source"].tolist()
    test_ds = CLFDataset(sents, tokenizer, \
                        max_len = config.maxlen, labels = df_test["label"].tolist())

    test_dl = DataLoader(test_ds, batch_size=config.per_device_batch_size, shuffle=False, num_workers=4)

    #### Get Model, Output Path
    model_name = config.pretrained_model.split("/")[-1]
    if config.fix_bert:
        model_name += "_fixbert"
    if config.label_smoothing_factor > 0:
        model_name += "_ls"

    # output_dir = str(os.path.join(args.output,model_name))
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    output_dir = args.output

    # No specified step
    if args.checkpoint_step < 0:
        model_path = str(os.path.join(args.model_dir,model_name+"/best"))
        output_path = os.path.join(output_dir, f'M{model_name}_best_F{file_name}.tsv')
        output_path4diff = os.path.join(output_dir, f'M{model_name}_best_F{file_name}-diff.tsv')
    else:
        model_path = str(os.path.join(args.model_dir,model_name+f"/checkpoint-{args.checkpoint_step}"))
        output_path = os.path.join(output_dir, f'M{model_name}_step-{args.checkpoint_step}_F{file_name}.tsv')
        output_path4diff = os.path.join(output_dir, f'M{model_name}_step-{args.checkpoint_step}_F{file_name}-diff.tsv')

    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(output_path4diff):
        os.remove(output_path4diff)

    print(model_path)
    #### Load Saved Model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    #### Prepare Output Files


    csvfile = open(output_path, "w", newline='')
    csvwriter = csv.writer(csvfile, delimiter='\t')

    csvfile4diff = open(output_path4diff, "w", newline='')
    csvwriter4diff = csv.writer(csvfile4diff, delimiter='\t')

    #### Evaluation
    y_target = []
    y_pred = []
    y_probs = []
    sentence_list = []

    # Softmax
    softmax = torch.nn.Softmax(dim=1)

    for batch in tqdm(test_dl):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            sentence = batch["sents"]

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            out = outputs.logits

            res = torch.argmax(out, axis=1) 

            y_target.extend(label.tolist())
            sentence_list.extend(sentence)
            y_pred.extend(res.tolist())

            # Also record probs
            y_probs.extend(softmax(out).tolist())

    #### Get Performance Measures
    accuracy = calc_acc(y_target, y_pred)
    precision, recall, f1score = calc_prf(y_target, y_pred, pos_label = 1, beta = 1.0)
    tn, fp, fn, tp = calc_conf_matrix(y_target, y_pred).ravel()

    ## Optional - ROC/MCC
    # Use probability of class 1 as score
    # auc = calc_roc(y_target, np.array(y_probs)[:,1], make_plot = False)
    # mcc = calc_mcc(y_target, y_pred)

    # Write to CSV
    csvwriter.writerow(["source", "label", "prediction"])
    csvwriter4diff.writerow(["source", "label", "prediction"])
    for i in range(len(y_target)):
        csvwriter.writerow([sentence_list[i], y_target[i], y_pred[i]])
        if y_target[i] != y_pred[i]:
            csvwriter4diff.writerow([sentence_list[i], y_target[i], y_pred[i]])

    csvwriter.writerow([f"LABEL"])
    csvwriter.writerow([f"POSITIVE= {tp+fn:,} NEGATIVE= {tn+fp:,}"])
    csvwriter.writerow([""])
    csvwriter.writerow([f"PREDICTION"])
    csvwriter.writerow([f"TP= {tp:,} FP= {fp:,}"])
    csvwriter.writerow([f"TN= {tn:,} FN= {fn:,}"])
    csvwriter.writerow([""])
    csvwriter.writerow(["SCORE"])
    csvwriter.writerow(["f1-score", f1score])
    csvwriter.writerow(["recall", recall])
    csvwriter.writerow(["precision", precision])
    csvwriter.writerow(["accuracy", accuracy])
    # csvwriter.writerow([f"Accuracy: {acc}"])
    # Additional Perf Metrics
    # csvwriter.writerow([f"AUC: {auc}"])
    # csvwriter.writerow([f"MCC: {mcc}"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='input file')
    parser.add_argument('model_dir', help='model path')
    parser.add_argument('output', help='output directory')
    parser.add_argument('--checkpoint_step', default=-1, type=int)
    parser.add_argument('--config_dir', default='config', help="Directory containing config.json of model")
    args = parser.parse_args()

    main(args)
