# https://towardsdatascience.com/transformers-retraining-roberta-base-using-the-roberta-mlm-procedure-7422160d5764


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="roberta-base",
    tokenizer="roberta-base"
)
fill_mask("Send these <mask> back!")

from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

#### Load Dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/tmp/tweeteval/datasets/hate/train_text.txt",
    block_size=512,
)


#### Define Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


#### Define Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./roberta-retrained",
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=48,
    save_steps=500,
    save_total_limit=2,
    seed=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


trainer.train()

trainer.save_model("./roberta-retrained")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./roberta-retrained",
    tokenizer="roberta-base"
)
fill_mask("Send these <mask> back!")