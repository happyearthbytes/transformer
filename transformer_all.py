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

from pprint import pprint, pformat
import inspect
from pygments import highlight
from pygments.lexers import Python3Lexer
from pygments.formatters import TerminalTrueColorFormatter

class Format:
    def __init__(self, style):
        self.style = style
    def print_py(self, code):
        print(highlight(code, Python3Lexer(), formatter=TerminalTrueColorFormatter(style=self.style)))
    def print_obj(self, item):
        self.print_py(pformat(item))
    def print_src(self, item):
        self.print_py(inspect.getsource(item))

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
    emb: int = 32
    device: str = "cuda"
    start_time: float = field(default_factory=lambda : time.time())
    last_time: float = 0.0

    def elapsed(self, label=""):
        elapsed_time = time.time() - self.start_time
        delta_time = elapsed_time - self.last_time if self.last_time > 0.0 else 0.0
        print(f"{elapsed_time:10.6f}s {delta_time:10.6f}s |\t{label}")
        self.last_time = elapsed_time

class Modules:
    class SelfAttention(torch.nn.Module):
        def __init__(self, emb, heads):
            super().__init__()
            assert emb % heads == 0
            self.emb, self.heads = emb, heads
            self.to_queries = torch.nn.Linear(emb, emb)
            self.to_keys = torch.nn.Linear(emb, emb)
            self.to_values = torch.nn.Linear(emb, emb)
            self.unify = torch.nn.Linear(emb, emb)

        def forward(self, x):
            b, t, emb = x.shape
            h = self.heads
            queries = self.to_queries(x)
            keys = self.to_keys(x)
            values = self.to_values(x)
            queries = queries.view(b, t, h, emb//h)
            keys = keys.view(b, t, h, emb//h)
            values = values.view(b, t, h, emb//h)
            queries = queries.transpose(1, 2).reshape(b*h, t, emb//h)
            keys = keys.transpose(1, 2).reshape(b*h, t, emb//h)
            values = values.transpose(1, 2).reshape(b*h, t, emb//h)
            W = torch.bmm(queries, keys.transpose(1,2))
            W = W / (emb**(1/2))
            W = F.softmax(W, dim=2)
            y = torch.bmm(W, values).view(b, h, t, emb//h)
            y = y.transpose(1, 2).reshape(b, t, emb)
            return self.unify(y), W

    class TransformerBlock(torch.nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.attention = Modules.SelfAttention(config.emb, config.heads)
            self.norm1 = torch.nn.LayerNorm(config.emb)
            self.norm2 = torch.nn.LayerNorm(config.emb)
            self.fcn = torch.nn.Sequential(
                torch.nn.Linear(config.emb, 4*config.emb),
                torch.nn.ReLU(),
                torch.nn.Linear(4*config.emb, config.emb)
            )

        def forward(self, x):
            attented, W = self.attention(x)
            x = self.norm1(attented + x)
            ff = self.fcn(x)
            return self.norm2(ff + x), W

    class Transformer(torch.nn.Module):
        def __init__(self, config: Config, vocab_size: int):
            super().__init__()
            self.config = config
            self.token_embedding = torch.nn.Embedding(embedding_dim=self.config.emb, num_embeddings=vocab_size)
            self.pos_embedding = torch.nn.Embedding(embedding_dim=self.config.emb, num_embeddings=self.config.max_length)
            self.tblock = Modules.TransformerBlock(config=self.config)
            self.toprobs = torch.nn.Linear(self.config.emb, 2)

        def forward(self, x):
            tokens = self.token_embedding(x)
            b, t, e = tokens.shape
            positions = self.pos_embedding(torch.arange(t, device=self.config.device))[None, :, :].expand(b, t, e)
            x = tokens + positions
            x, W = self.tblock(x)
            x = torch.mean(x, dim=1)
            x = self.toprobs(x)
            return F.log_softmax(x, dim=1), W

class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.config.elapsed("DataHandler Start")
        self.dataset = datasets.load_dataset("zapsdcn/imdb", cache_dir="../data/classification/")
        self.config.elapsed("DataHandler load_dataset")
        self.train_dataset = self.dataset['train'] 
        self.validation_dataset = self.dataset['validation']
        self.test_dataset = self.dataset['test']
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.config.elapsed("DataHandler get_tokenizer")
        self.trim = slice(None,config.max_length,None)
        self.tokenize_all()
        self.config.elapsed("DataHandler tokenize_all")
        self.set_vocab()
        self.config.elapsed("DataHandler set_vocab")
        self.numericalize_all()
        self.config.elapsed("DataHandler numericalize_all")
        self.format_all()
        self.config.elapsed("DataHandler format_all")
        self.set_data_loaders()
        self.config.elapsed("DataHandler set_data_loaders")

    def tokenize(self, data):
        tokens = self.tokenizer(data["text"])[self.trim]
        return {"tokens": tokens}
    
    def numericalize(self, data):
        id = self.vocab.lookup_indices(data["tokens"])
        return {"id": id}

    def tokenize_all(self):
        self.train_dataset = self.train_dataset.map(self.tokenize)
        self.validation_dataset = self.validation_dataset.map(self.tokenize)
        self.test_dataset = self.test_dataset.map(self.tokenize)

    def numericalize_all(self):
        self.train_dataset = self.train_dataset.map(self.numericalize)
        self.validation_dataset = self.validation_dataset.map(self.numericalize)
        self.test_dataset = self.test_dataset.map(self.numericalize)

    def format_all(self):
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



class TrainHandler:
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True

    def load(self, data_handler: DataHandler):
        self.config.elapsed("TrainHandler start")
        self.data_handler = data_handler
        self.model = Modules.Transformer(config=self.config, vocab_size=len(self.data_handler.vocab)).to(self.config.device)
        self.opt = torch.optim.Adam(lr=self.config.lr, params=self.model.parameters())
        self.config.elapsed("TrainHandler load")

    def train(self):
        self.config.elapsed("train start")
        self.accs = []
        for epoch in range(self.config.num_epochs):
            for batch in self.data_handler.train_data_loader:
                self.config.elapsed(f"train batch start")
                self.opt.zero_grad()
                input = batch["id"].to(self.config.device)
                output = batch["label"].to(self.config.device)
                self.config.elapsed("train batch ready")
                preds, _ = self.model(input)
                self.config.elapsed("train batch pred")
                loss = F.nll_loss(preds, output)
                loss.backward()
                self.opt.step()
                self.config.elapsed("train batch step")
                with torch.no_grad():
                    tot, cor= 0.0, 0.0
                    for batch in self.data_handler.validation_data_loader:
                        self.config.elapsed("train validate start")
                        input = batch["id"].to(self.config.device)
                        output = batch["label"].to(self.config.device)
                        if input.shape[1] > self.config.max_length:
                            input = input[:, :self.config.max_length]
                        self.config.elapsed("train validate ready")
                        preds, _ = self.model(input)
                        self.config.elapsed("train validate pred")
                        preds = preds.argmax(dim=1)
                        tot += float(input.size(0))
                        cor += float((output == preds).sum().item())
                        self.config.elapsed("train validate done")
                    acc = cor / tot
                    self.accs.append(acc)
                self.config.elapsed("train batch done")
            print("Epoch:{}; Loss: {}; Validation Accuracy: {}".format(epoch, loss.item(), acc))

    def save(self):
        torch.save(self.model.state_dict(), "trained_models/clasify_{}heads.pt".format(self.config.heads))
        np.save("trained_models/acc.npy", self.accs)
        self.config.elapsed("save")

def main():
    config=Config()
    train_handler = TrainHandler(config=config)
    data_handler = DataHandler(config=config)
    train_handler.load(data_handler=data_handler)
    train_handler.train()
    train_handler.save()

if __name__ == "__main__":
    main()