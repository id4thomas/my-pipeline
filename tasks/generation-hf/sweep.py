import argparse
import wandb
import json

from train_hf_dec import train as train_dec
from train_hf_encdec import train as train_encdec

# https://stackoverflow.com/questions/37367331/is-it-possible-to-use-argparse-to-capture-an-arbitrary-set-of-optional-arguments
if __name__ == '__main__':
    # Parse Args
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            # parser.add_argument(arg.split('=')[0], type=<your type>, ...)
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    with wandb.init(config = args) as run:
        run_config = wandb.config

        # Load Train Config
        with open(run_config["config_dir"], 'r') as f:
            config = json.loads(f.read())
        # Apply sweep params to train_config
        config.update(run_config)
        
        train_encdec(config, save_trained = False)
