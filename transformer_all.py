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

class Const:
    seed=0 # use fixed seed for testing
    preview = slice(None,10,None) # first 10 items
    example_item = 1337
    max_length = 512 # trim sequences
    min_freq = 5 # filter uncommon tokens
    special_tokens = ["<unk>", "<pad>"]
    batch_size = 512
    lr = 0.001
    num_epochs = 20
    heads = 4

class TrainHandler:
    @staticmethod
    def config(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True

class DataHandler:
    def __init__(self):
        dataset = datasets.load_dataset("zapsdcn/imdb", cache_dir="../data/classification/")
        self.train_dataset = dataset['train'] 
        self.validation_dataset = dataset['validation']
        self.test_dataset = dataset['test']
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def main():
    print("B")

if __name__ == "__main__":
    main()