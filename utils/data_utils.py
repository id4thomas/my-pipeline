from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import Dataset
import numpy as np

class CLFDataset(Dataset):
    def __init__(self, sents, tokenizer, max_len, labels = None, is_test = False, padding = "max_length", truncation = True):

        self.encodings = tokenizer(sents, padding = padding, max_length = max_len, truncation = truncation, return_tensors="pt")
        self.is_test = is_test
        self.sents = sents
        if not self.is_test:
            self.labels = labels


    def __len__(self) -> int:
        return (len(self.sents))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["sents"] = self.sents[idx]
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx])
        return item