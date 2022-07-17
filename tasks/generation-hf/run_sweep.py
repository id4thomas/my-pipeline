import json
import yaml
import os
import argparse
import wandb 

def find_prefix(dirpath):
    file_list = os.listdir(dirpath)
    prefix_list = [file.replace('-train.tsv', '') for file in file_list if file.endswith('-train.tsv')]
    return prefix_list

parser = argparse.ArgumentParser()
parser.add_argument('project_name')
parser.add_argument('--data_dir', default='corpus/train')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--train_config_dir', default='config/config.json')
parser.add_argument('--sweep_config_dir', default='config/sweep.json')
parser.add_argument('--count', default=5, type=int)
args = parser.parse_args()

prefixes = find_prefix(args.data_dir)
if len(prefixes) == 0:
    raise AssertionError("no label data")

# Get Sweep Config
with open(args.sweep_config_dir, 'r') as f:
    sweep_config = json.loads(f.read())

sweep_config["program"] = "sweep.py"

sweep_config["parameters"]["config_dir"] = {'value':args.train_config_dir}
sweep_config["parameters"]["data_dir"] = {'value':args.data_dir}
sweep_config["parameters"]["model_dir"] = {'value':args.model_dir}
sweep_config["parameters"]["data_prefix"] ={'value': prefixes[0]}
sweep_config["parameters"]["project_name"] = {'value':args.project_name}

# with open(os.path.join(args.config_dir,'sweep_test.yaml'), 'w') as f:
#     yaml.safe_dump(sweep_config, f, allow_unicode = True)

sweep_id = wandb.sweep(sweep_config, project = args.project_name, entity = "id4thomas")
print(sweep_id)

os.system(f"wandb agent id4thomas/{args.project_name}/{sweep_id} --count {args.count}")