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

class Config:
    seed=0 # use fixed seed for testing
    preview = slice(None,10,None) # first 10 items
    example_item = 1337
    max_length = 512 # trim sequences
    min_freq = 5 # filter uncommon tokens
    default_token = "<unk>"
    special_tokens = [default_token, "<pad>"]
    batch_size = 512
    lr = 0.001
    num_epochs = 20
    heads = 4

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
        self.datasets = ("train","validation","test")
        self.train_dataset = self.dataset['train'] 
        self.validation_dataset = self.dataset['validation']
        self.test_dataset = self.dataset['test']
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.trim = slice(None,config.max_length,None)
        self.tokenize_all()
        self.set_vocab()
        self.numericalize_all()
        self.format_all()

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