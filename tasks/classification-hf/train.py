#-*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import random
import os
import sys
from pathlib import Path
import traceback

# third party 
import numpy as np
import pandas as pd
import wandb

# torch
import torch

# transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


# Utilities
from utils.data_utils import CLFDataset
from utils.perf_utils import calc_acc, calc_prf
from utils.utils import Config

logger = logging.getLogger(__name__)

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def wandb_set(args, data_prefix, config, run_name):
    run = wandb.init(project=args.project_name, entity='id4thomas', config=config, name = run_name)

    artifact1 = wandb.Artifact(args.project_name + '-dataset', type='dataset')
    artifact1.add_file(args.data_dir + '/' + data_prefix + '-train.tsv')
    artifact1.add_file(args.data_dir + '/' + data_prefix + '-val.tsv')
    run.log_artifact(artifact1)

    artifact2 = wandb.Artifact('config', type='config')
    artifact2.add_file(args.config_dir + '/config.json')
    run.log_artifact(artifact2)

# compute_metrics for Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1 = calc_prf(labels, preds, average='binary', beta = 1.0)
    acc = calc_acc(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def run(args, prefix_list):
    #### Config
    config_dir = Path(args.config_dir)
    config = Config(json_path=config_dir / 'config.json')

    set_seed(config.seed)

    #### Load Data
    data_prefix = prefix_list[0]
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    # Train & Val Datasets
    train_fpath = str(os.path.join(args.data_dir,data_prefix+"-train.tsv"))
    val_fpath = str(os.path.join(args.data_dir,data_prefix+"-val.tsv"))
    
    df_train = pd.read_csv(str(train_fpath), sep="\t")
    df_val = pd.read_csv(str(val_fpath), sep="\t")

    tr_ds = CLFDataset(df_train["source"].tolist(), tokenizer, \
                        max_len = config.maxlen, labels = df_train["label"].tolist())
    val_ds = CLFDataset(df_val["source"].tolist(), tokenizer, \
                        max_len = config.maxlen, labels = df_val["label"].tolist())

    #### No need for dataloader with Trainer
    # tr_dl = DataLoader(tr_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=False)
    # val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=False)


    #### Model
    ## model_name: used in directories
    ## run_name: used for tracking in wandb
    model_name = config.pretrained_model.split("/")[-1]
    run_name = f"{config.pretrained_model}_ep{config.epochs}_lr{config.learning_rate}"

    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model, num_labels = config.num_labels)
    
    # Additional Options - follow trainer args if possible
    if config.fix_bert:
        # Fix weights of BERT portion of model
        for param in model.roberta.encoder.parameters():
            param.requires_grad = False
        run_name+="_fixbert"
        model_name += "_fixbert"
    if config.label_smoothing_factor > 0:
        run_name+="_ls"
        model_name += "_ls"

    #### Prepare Directories
    out_dir = os.path.join(args.model_dir, f"{model_name}")

    checkpoint_dir = os.path.abspath(out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_dir = Path(out_dir)
    print("model_dir : ", model_dir)
    
    #### Init Wandb
    wandb_set(args, data_prefix, config, run_name) # 데이터 로딩이 끝난 후에, wandb 연결
    wandb.watch(model, log="all", log_freq=10)

    effective_batch_size = config.per_device_batch_size*config.gradient_accumulation_steps*torch.cuda.device_count()
    print("Effective Batch Size:",effective_batch_size)

    # Trainer Based Training
    training_args = TrainingArguments(
        run_name = run_name,
        # Train Params
        seed = config.seed,
        num_train_epochs = config.epochs,
        label_smoothing_factor = config.label_smoothing_factor,
        per_device_train_batch_size = config.per_device_batch_size,
        per_device_eval_batch_size = config.per_device_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        learning_rate = config.learning_rate,
        # Checkpointing, Saving
        output_dir = os.path.join(out_dir,"checkpoints"),
        save_strategy = "steps", # steps, epoch
        save_steps = config.save_steps,
        load_best_model_at_end = True,
        # Evaluating, Logging
        evaluation_strategy = "steps", #steps, epoch, no
        logging_dir = out_dir,
        logging_steps = config.summary_step,
        disable_tqdm = False,
        report_to = "wandb", # "azure_ml", "comet_ml", "mlflow", "tensorboard", "wandb"
        # System
        fp16 = config.fp16,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tr_ds,
        eval_dataset = val_ds,
        compute_metrics = compute_metrics,
        #optimizers = # Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional)
    )

    trainer.train()

    #### Save Best Model
    best_dir = os.path.join(out_dir,"best")
    trainer.save_model(str(best_dir))


def find_prefix(dirpath):
    file_list = os.listdir(dirpath)
    prefix_list = [file.replace('-train.tsv', '') for file in file_list if file.endswith('-train.tsv')]
    return prefix_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--data_dir', default='data/train')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--config_dir', default='config.json')

    args = parser.parse_args()

    prefixes = find_prefix(args.data_dir)
    if len(prefixes) == 0:
        raise AssertionError("no label data")

    try:
        run(args, prefixes)
    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)

