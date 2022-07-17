TASK=ENCDEC

DATA_DIR=data
DATA_NAME=data1

MODEL_DIR=models/$DATA_NAME
CONFIG_DIR=config/$DATA_NAME/train.json

python train_hf_encdec.py ${TASK} --model_dir=${MODEL_DIR} --data_dir=${DATA_DIR} --config_dir=${CONFIG_DIR}