# Example Sweep Script
TASK=sweep_test
DATA_DIR=data

DATA_NAME=data1
MODEL_DIR=models_sweep/$DATA_NAME
SWEEP_CONFIG_DIR=config/$DATA_NAME/sweep.json
TRAIN_CONFIG_DIR=config/$DATA_NAME/base.json

python run_sweep.py ${TASK} \
        --model_dir=${MODEL_DIR} \
        --data_dir=${DATA_DIR} \
        --train_config_dir=${TRAIN_CONFIG_DIR} \
        --sweep_config_dir=${SWEEP_CONFIG_DIR} \
        --count 15

rm -rf models_sweep/$DATA_NAME/kobart