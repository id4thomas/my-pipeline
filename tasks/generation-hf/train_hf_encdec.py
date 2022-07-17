#-*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import logging
import random
import os
import sys
import time
from pathlib import Path
import pandas as pd
import wandb
import traceback
import json
import csv

# third party 
# torch
import torch

# transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset

# Utilities
from utils.data_utils import *
from utils.eval_utils import *
# from utils.utils import Config

logger = logging.getLogger(__name__)

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def wandb_set(project_name, data_dir, data_prefix, config_dir, run_name):
    # run = wandb.init(project=project_name, entity='mauritia', config=config, name = run_name)
    run = wandb.init(project=project_name, entity='id4thomas', config=config, name = run_name)

    artifact1 = wandb.Artifact(project_name + '-dataset', type='dataset')
    artifact1.add_file(data_dir + '/' + data_prefix + '-train.tsv')
    artifact1.add_file(data_dir + '/' + data_prefix + '-val.tsv')
    run.log_artifact(artifact1)

    artifact2 = wandb.Artifact('config', type='config')
    artifact2.add_file(config_dir)
    run.log_artifact(artifact2)


def train(config, save_trained = True):
    #### Set Seed
    set_seed(config["seed"])

    #### Load Data
    data_prefix = config["data_prefix"]
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
    # num_added_toks = tokenizer.add_tokens(['[EOT]'], special_tokens=True)
    num_added_toks = tokenizer.add_tokens(['<hl>', '</hl>'])

    # Train & Val Datasets
    train_fpath = str(os.path.join(config["data_dir"],data_prefix+"-train.tsv"))
    val_fpath = str(os.path.join(config["data_dir"],data_prefix+"-val.tsv"))

    df_train = pd.read_csv(str(train_fpath), sep="\t", quoting=csv.QUOTE_NONNUMERIC)
    df_val = pd.read_csv(str(val_fpath), sep="\t", quoting=csv.QUOTE_NONNUMERIC)

    # tr_ds = GenEncDecDataset(df_train["source"].tolist(), tokenizer, \
    #                     max_len = config.maxlen, labels = df_train["label"].tolist())
    # val_ds = GenEncDecDataset(df_val["source"].tolist(), tokenizer, \
    #                     max_len = config.maxlen, labels = df_val["label"].tolist())

    # tr_ds = load_dataset('csv', data_files = train_fpath)
    # val_ds = load_dataset('csv', data_files = val_fpath)
    tr_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    print(tr_ds.info)

    # exit()
    tr_ds = tr_ds.map(
        lambda batch: batch_tokenize_preprocess_encdec(
            batch, tokenizer, config["encoder_max_length"], config["decoder_max_length"]
        ),
        batched=True,
        # remove_columns=[""],
    )

    val_ds = val_ds.map(
        lambda batch: batch_tokenize_preprocess_encdec(
            batch, tokenizer, config["encoder_max_length"], config["decoder_max_length"]
        ),
        batched=True,
        # remove_columns=[""],
    )

    #### No need for dataloader with Trainer
    # tr_dl : 데이터셋을 batch_size 만큼씩 나누어 둠
    # tr_dl = DataLoader(tr_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=False)
    # val_dl : 데이터셋을 batch_size 만큼씩 나누어 둠
    # val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=False)


    #### Model
    ## model_name: used in directories
    ## run_name: used for tracking in wandb
    model_name = config["pretrained_model"].split("/")[-1]
    run_name = "{}_ep{}_lr{}".format(config["pretrained_model"], config["epochs"], config["learning_rate"])

    model = AutoModelForSeq2SeqLM.from_pretrained(config["pretrained_model"])
    model.resize_token_embeddings(len(tokenizer))
    
    #### Prepare Directories
    out_dir = os.path.join(config["model_dir"], f"{model_name}")

    checkpoint_dir = os.path.abspath(out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_dir = Path(out_dir)
    print("model_dir : ", model_dir)
    
    #### Init Wandb
    # wandb_set(args, data_prefix, config, run_name) # 데이터 로딩이 끝난 후에, wandb 연결
    wandb.watch(model, log="all", log_freq=10)

    effective_batch_size = config["per_device_batch_size"]*config["gradient_accumulation_steps"]*torch.cuda.device_count()
    print("Effective Batch Size:",effective_batch_size)

    # Trainer Based Training
    training_args = Seq2SeqTrainingArguments(
        run_name = run_name,

        # Train Params
        ## Steps/Epochs
        num_train_epochs = config["epochs"],
        # max_steps = 2,

        ## LR
        learning_rate = config["learning_rate"],
        ## Batch
        per_device_train_batch_size = config["per_device_batch_size"],
        per_device_eval_batch_size = config["per_device_batch_size"],
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        ## ETC
        label_smoothing_factor = config["label_smoothing_factor"],

        # Checkpointing, Saving
        output_dir = os.path.join(out_dir,"checkpoints"),
        save_strategy = "steps", # steps, epoch
        save_steps = config["save_steps"],
        save_total_limit = config["save_total_limit"],
        load_best_model_at_end = True,
        overwrite_output_dir=True,

        # Evaluating
        evaluation_strategy = "steps",
        metric_for_best_model = config["metric_for_best_model"],

        # Logging
        logging_dir = out_dir,
        logging_steps = config["summary_step"],
        disable_tqdm = False,
        report_to = "wandb",
        predict_with_generate = True,

        # System
        seed = config["seed"],
        fp16 = config["fp16"],
        bf16 = config["bf16"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator=data_collator,
        train_dataset = tr_ds,
        eval_dataset = val_ds,
        # compute_metrics = compute_metrics,
        compute_metrics = GenEvaluator(tokenizer),
        #optimizers = ,# Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional)
    )

    trainer.train()
    trainer.evaluate(val_ds, metric_key_prefix = "final")
    if save_trained:
        # #### Save Best Model
        best_dir = os.path.join(out_dir,"best")
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        with open(os.path.join(best_dir, "train_configs.json"), 'w') as f:
            json.dump(config, f)

    # # Also to binaries/ linked directory
    # best_dir = os.path.join(args.model_dir, f"model/{model_name}")
    # trainer.save_model(str(best_dir))
    # tokenizer.save_pretrained(str(best_dir))


def find_prefix(dirpath):
    file_list = os.listdir(dirpath)
    prefix_list = [file.replace('-train.tsv', '') for file in file_list if file.endswith('-train.tsv')]
    return prefix_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--data_dir', default='corpus/train')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--config_dir', default='config')

    args = parser.parse_args()

    # Get Data prefix
    prefixes = find_prefix(args.data_dir)
    if len(prefixes) == 0:
        raise AssertionError("no label data")
    prefix = prefixes[0]

    # Load Config
    with open(args.config_dir, 'r') as f:
        config = json.loads(f.read())

    config.update(vars(args))
    config["data_prefix"] = prefix

    # WANDB Init
    run_name = "{}_ep{}_lr{}".format(config["pretrained_model"], config["epochs"], config["learning_rate"])

    if config["label_smoothing_factor"] > 0:
        run_name+="_ls"
    wandb_set(args.project_name, args.data_dir, prefix, args.config_dir, run_name)

    try:
        train(config, save_trained = True)
    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)

