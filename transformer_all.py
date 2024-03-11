#!/usr/bin/env python

import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from pprint import pprint
import time
from functools import partial
from dataclasses import dataclass, field

@dataclass
class Config:
    seed: int = 0 # use fixed seed for testing
    preview: slice = field(default_factory=lambda : slice(None,10,None))  # first N items
    example_item: int = 1337
    max_length: int = 512 # trim sequences
    min_freq: int = 5 # filter uncommon tokens
    default_token: str = "<unk>"
    special_tokens: list[str] = field(default_factory=lambda : ["<unk>", "<pad>"])
    batch_size: int = 512
    lr: float = 0.001
    num_epochs: int = 20
    heads: int = 4

class TrainHandler:
    @staticmethod
    def config(config: Config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True

class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = datasets.load_dataset("zapsdcn/imdb", cache_dir="../data/classification/")
        self.train_dataset = self.dataset['train'] 
        self.validation_dataset = self.dataset['validation']
        self.test_dataset = self.dataset['test']
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.trim = slice(None,config.max_length,None)
        self.tokenize_all()
        self.set_vocab()
        self.numericalize_all()
        self.format_all()
        self.set_data_loaders()

    def tokenize(self, data):
        tokens = self.tokenizer(data["text"])[self.trim]
        return {"tokens": tokens}
    
    def numericalize(self, data):
        id = self.vocab.lookup_indices(data["tokens"])
        return {"id": id}

    def tokenize_all(self):
        # for dataset in self.datasets:
        #     self.dataset[dataset] = self.dataset[dataset].map(self.tokenize)
        self.train_dataset = self.train_dataset.map(self.tokenize)
        self.validation_dataset = self.validation_dataset.map(self.tokenize)
        self.test_dataset = self.test_dataset.map(self.tokenize)

    def numericalize_all(self):
        # for dataset in self.datasets:
        #     self.dataset[dataset] = self.dataset[dataset].map(self.numericalize)
        self.train_dataset = self.train_dataset.map(self.numericalize)
        self.validation_dataset = self.validation_dataset.map(self.numericalize)
        self.test_dataset = self.test_dataset.map(self.numericalize)

    def format_all(self):
        # for dataset in self.datasets:
        #     self.dataset[dataset] = self.dataset[dataset].with_format(type="torch", columns=["id", "label"])
        self.train_dataset = self.train_dataset.with_format(type="torch", columns=["id", "label"])
        self.validation_dataset = self.validation_dataset.with_format(type="torch", columns=["id", "label"])
        self.test_dataset = self.test_dataset.with_format(type="torch", columns=["id", "label"])
        
    def set_vocab(self):
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.train_dataset["tokens"],
            min_freq=self.config.min_freq,
            specials=self.config.special_tokens,
        )
        self.vocab.set_default_index(self.vocab[self.config.default_token])

    @staticmethod
    def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
        def get_collate_fn(pad_index):
            def collate_fn(batch):
                batch_ids = [i["id"] for i in batch]
                batch_ids = torch.nn.utils.rnn.pad_sequence(
                    batch_ids, padding_value=pad_index, batch_first=True
                )
                batch_label = [i["label"] for i in batch]
                batch_label = torch.stack(batch_label)
                batch = {"id": batch_ids, "label": batch_label}
                return batch
            return collate_fn

        collate_fn = get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader

    def set_data_loaders(self):
        pad_index = self.vocab["<pad>"]
        self.train_data_loader = DataHandler.get_data_loader(self.train_dataset, self.config.batch_size, pad_index, shuffle=True)
        self.validation_data_loader = DataHandler.get_data_loader(self.validation_dataset, self.config.batch_size, pad_index)
        self.test_data_loader = DataHandler.get_data_loader(self.test_dataset, self.config.batch_size, pad_index)


def elapsed_from_start(start_time, label=""):
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:10.6f}s |\t{label}")

def main():
    start_time = time.time()
    elapsed = partial(elapsed_from_start,start_time)
    config=Config()
    TrainHandler.config(config=config)
    elapsed("Config")
    dh = DataHandler(config=config)
    elapsed("DataHandler")
    end_time = time.time()
    elapsed("End")


if __name__ == "__main__":
    main()