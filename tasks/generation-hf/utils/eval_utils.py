from datasets import Metric, load_metric
from torch.utils.data import DataLoader

from KoBERTScore import BERTScore

import wandb
import numpy as np

from transformers import TrainerCallback

# Similar approach to 
# https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/src/evaluation/seq_to_seq.py

class GenEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.normal_bleu = load_metric("bleu")
        self.smooth_bleu = False
        self.rouge = load_metric("rouge")
        self.bertscore = BERTScore("beomi/kcbert-base", best_layer=4)

    # When called through compute_metrics of Seq2SeqTrainer
    def __call__(self, preds):
        preds, labels = preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return self.evaluate(decoded_preds, decoded_labels)
        
    def calcBLEU(self, decoded_preds, decoded_labels):
        # Calculate the BLEU scores then return them.
        def bleuTok(arr):
            return list(map(lambda x: x.split(' '), arr))

        bleu_toked_preds = bleuTok(decoded_preds)
        blue_toked_labels = [[x] for x in bleuTok(decoded_labels)]
        return self.normal_bleu.compute(
            predictions=bleu_toked_preds,
            references=blue_toked_labels,
            smooth=self.smooth_bleu
        )
    def evaluate(self, decoded_preds, decoded_labels):
        print(decoded_preds, decoded_labels)
        bleu_scores = self.calcBLEU(decoded_preds, decoded_labels)
        rogue_scores = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)

        # High rouge
        # rouge_l = rogue_scores['rougeL'].high
        # Mid rouge
        rouge_l = rogue_scores['rougeL'].mid

        # __call__ of BERTScore only returns F
        bertscore  = self.bertscore(decoded_labels, decoded_preds, batch_size=128)

        return {
            "BLEU": bleu_scores['bleu'] * 100,
            "ROUGE-L": rouge_l.fmeasure * 100,
            "BERTSCORE": sum(bertscore)/len(bertscore) # Just return average
        }

# For Dec Models - Custom Callback
# add to trainer by trainer.add_callback(DecGenEvalCallback)
class DecGenEvalCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.normal_bleu = load_metric("bleu")
        self.smooth_bleu = False
        self.rouge = load_metric("rouge")
        self.bertscore = BERTScore("beomi/kcbert-base", best_layer=4)

    # Called on evalute step
    def on_evaluate(self, args, state, control, **kwargs):
        # Receives model, tokenizer, train_dataloader, eval_dataloader, metrics (values)

        model.eval()
        
        # Generate (Greedy)
        decode_params = {
            'do_sample': False,
            'early_stopping': True,
            'temperature': 1.0,
            'repetition_penalty': 1.0
            # 'max_length': 
        }
        pad_token_id = tokenizer.pad_token_id

        # Prepare DataLoader
        val_dataset = eval_dataloader.dataset
        val_collator = lambda data: {
            'input_ids': torch.cat([tokenizer(f['source'], return_tensors="pt", truncation=True, max_length=128, padding="max_length")["input_ids"] for f in data], dim=0),
            'attention_mask': torch.cat([tokenizer(f['source'], return_tensors="pt", truncation=True, max_length=128, padding="max_length")["attention_mask"] for f in data], dim=0),
            'source': [f['source'] for f in data],
            'target': [f['target'] for f in data],
        }

        val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, collate_fn=val_collator)

        preds = []
        answers = []
        for ii, batch in enumerate(tqdm(val_loader)):
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)

            with torch.no_grad():
                outs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_generate=1,
                    pad_token_id=tokenizer.pad_token_id,
                    **decode_params
                )
                
                for input_id, out in zip(input_ids, outs):
                    input_len = input_id[input_id != pad_token_id].shape[0]
                    gen = tokenizer.decode(out[input_len:], skip_special_tokens = True)
                    preds.append(gen)

            answers.extend(batch["target"])

        # Calculate 
        bleu_scores = self.calcBLEU(preds, answers)
        rogue_scores = self.rouge.compute(predictions=preds, references=answers)

        # High rouge
        # rouge_l = rogue_scores['rougeL'].high
        # Mid rouge
        rouge_l = rogue_scores['rougeL'].mid

        # __call__ of BERTScore only returns F
        bertscore  = self.bertscore(answers, preds, batch_size=128)

        score_dict = {
            "val-BLEU": bleu_scores['bleu'] * 100,
            "val-ROUGE-L": rouge_l.fmeasure * 100,
            "val-BERTSCORE": sum(bertscore)/len(bertscore) # Just return average
        }
        wandb.log(score_dict)

# Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py

# nltk.download("punkt", quiet=True)

# metric = datasets.load_metric("rouge")


# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [label.strip() for label in labels]

#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

#     return preds, labels


# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Some simple post-processing
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(
#         predictions=decoded_preds, references=decoded_labels, use_stemmer=True
#     )
#     # Extract a few results from ROUGE
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

#     prediction_lens = [
#         np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
#     ]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result
