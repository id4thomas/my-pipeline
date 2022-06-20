from __future__ import absolute_import, division, print_function, unicode_literals

import torch
# from torch.utils.data import Dataset
import numpy as np

def batch_tokenize_preprocess_encdec(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["source"], batch["target"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss (-100 is ignored)
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

# GPT2 응용 질문
def batch_tokenize_preprocess_dec(batch, tokenizer, max_length):
    source, target = batch["source"], batch["target"]

    # For GPT-2
    tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', 
                                 truncation=True, 
                                 max_length=max_length, 
                                 padding="max_length")
      
    # source_tokenized = tokenizer(
    #     source, padding="max_length", truncation=True, max_length=max_source_length
    # )
    # target_tokenized = tokenizer(
    #     target, padding="max_length", truncation=True, max_length=max_target_length
    # )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss (-100 is ignored)
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

# GPT2 단순 질문 - continuation (cont)
def batch_tokenize_preprocess_dec_cont(batch, tokenizer, max_length):
    source, target = batch["source"], batch["target"]

    # For GPT-2
    
    input_sents = ['<|startoftext|>'+  s + '<|endoftext|><|startoftext|>' + t + '<|endoftext|>' for s,t in zip(source, target)]
    tokenized = tokenizer(input_sents, 
                                 truncation=True, 
                                 max_length=max_length, 
                                 padding="max_length")
      
    batch = {k: v for k, v in tokenized.items()}

    # Ignore padding in the loss (-100 is ignored) - Masking
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in tokenized["input_ids"]
    ]

    # Sentence too
    batch["source"] = source
    batch["target"] = target
    return batch

# Dataset for generation with Decoder model
# class GenDecDataset(Dataset):
#     def __init__(self, sents, tokenizer, max_len, labels = None, is_test = False, padding = "max_length", truncation = True):

#         self.encodings = tokenizer(sents, padding = padding, max_length = max_len, truncation = truncation, return_tensors="pt")
#         self.is_test = is_test
#         self.sents = sents
#         if not self.is_test:
#             self.labels = labels


#     def __len__(self) -> int:
#         return (len(self.sents))

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item["sents"] = self.sents[idx]
#         if not self.is_test:
#             item['labels'] = torch.tensor(self.labels[idx])
#         return item

#         if self.is_predict:
#             return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
#                     'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"])}
#         else:
#             return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
#                     'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"]),
#                     'labels': torch.tensor(self.labels[idx], dtype=torch.long)}


# # Dataset for generation with Encoder-Decoder model
# class GenEncDecDataset(Dataset):
#     def __init__(self, tokenizer, source, target, max_source_len, max_target_len):

#         self.source_tokenized = tokenizer(source, max_length = max_source_len, truncation = True, return_tensors="pt")
#         self.target_tokenized = tokenizer(target, max_length = max_target_len, truncation = True, return_tensors="pt")
#         self.target_input_ids = [
#             [-100 if token == tokenizer.pad_token_id else token for token in l]
#             for l in self.target_tokenized["input_ids"]
#         ]

#         self.is_test = is_test
#         self.sents = sents
#         if not self.is_test:
#             self.labels = labels


#     def __len__(self) -> int:
#         return (len(self.sents))

#     def __getitem__(self, idx):
#         # Source
#         item = {key: torch.tensor(val[idx]) for key, val in self.source_tokenized.items()}
#         # Target
#         # item["labels"] = 

#         item["sents"] = self.sents[idx]
#         if not self.is_test:
#             item['labels'] = torch.tensor(self.labels[idx])
#         return item

#         if self.is_predict:
#             return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
#                     'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"])}
#         else:
#             return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
#                     'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"]),
#                     'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

