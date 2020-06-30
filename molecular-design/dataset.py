from copy import deepcopy
from typing import Dict, List, Tuple, Union
import os
from collections import defaultdict

import numpy as np
import torch

PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNKNOWN_TOKEN = '<unk>'


class Dataset:
    def indices2smiles(self, indices: List[int]) -> str:
        if self.end_index in indices:
            indices = indices[1:indices.index(self.end_index)]
        else:
            indices = indices[1:]
        smiles = ''.join(self.np_i2s[indices])

        if len(smiles) == 0:
            smiles = PAD_TOKEN  # just feed it some garbage so at least chemprop won't crash on an empty smiles

        return smiles

    def smiles2indices(self, smiles: str) -> List[int]:

        # deal with special tokens; in practice this is only needed for --no_filter argument
        smiles_list = []
        in_special_tok = False
        for c in smiles:
            if in_special_tok:
                current_tok += c
                if c == '>':
                    smiles_list.append(current_tok)
                    in_special_tok = False
            else:
                if c == '<':
                    in_special_tok = True
                    current_tok = c
                else:
                    smiles_list.append(c)
        indices = [self.start_index] + [self.s2i[char] for char in smiles_list] + [self.end_index]

        return indices
    
    def set_batch_size(self, batch_size):
        self.batch_size = int(batch_size)


class PairDataset(Dataset):
    def __init__(self,
                 path: str,
                 i2s: List[str] = None,
                 batch_size: int = 32,
                 max_length: int = 100,
                 extra_vocab_path: str=None,
                 max_data: int=None):
        self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN = PAD_TOKEN, START_TOKEN, END_TOKEN

        self.src = []
        self.tgt = []
        self.src_smiles = []
        self.tgt_smiles = []

        if type(path) == str:
            f = open(path, 'r')
        else:
            f = path # gives option to directly pass in pairs as an array
        for line in f:
            line = line.strip().split(' ')
            assert len(line) == 2
            self.src.append(list(line[0][:max_length]))
            self.src_smiles.append(line[0][:max_length])
            self.tgt.append(list(line[1][:max_length]))
            self.tgt_smiles.append(line[1][:max_length])
        if type(path) == str:
            f.close()

        assert len(self.src) == len(self.tgt)

        vocab = set()
        for line in self.src:
            vocab.update(line)
        for line in self.tgt:
            vocab.update(line)

        if extra_vocab_path is not None:
            with open(extra_vocab_path, 'r') as rf:
                for line in rf:
                    line = list(line.strip())
                    vocab.update(line)

        if i2s is None:
            self.i2s = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
            self.i2s += sorted(vocab)
        else:
            self.i2s = i2s
            for c in sorted(vocab):
                if c not in i2s:
                    i2s.append(c)
        self.np_i2s = np.array(self.i2s)

        self.s2i = defaultdict(lambda: UNKNOWN_TOKEN)
        for i, s in enumerate(self.i2s):
            self.s2i[s] = i
        self.pad_index = self.s2i[self.PAD_TOKEN]  # 0
        self.start_index = self.s2i[self.START_TOKEN]  # 1
        self.end_index = self.s2i[self.END_TOKEN]  # 2

        for i in range(len(self.src)):
            self.src[i] = [self.start_index] + [self.s2i[char] for char in self.src[i]] + [self.end_index]
        for i in range(len(self.tgt)):
            self.tgt[i] = [self.start_index] + [self.s2i[char] for char in self.tgt[i]] + [self.end_index]

        self.src = np.array(self.src) # this is screwy if they're all the same length, np won't make it a list of lists
        self.tgt = np.array(self.tgt)
        self.pos = 0
        self.batch_size = batch_size

        if max_data is not None:
            self.reshuffle()
            self.src = self.src[:max_data]
            self.tgt = self.tgt[:max_data]
            self.src_smiles = self.src_smiles[:max_data]
            self.tgt_smiles = self.tgt_smiles[:max_data]

    def __len__(self) -> int:
        return len(self.src)
    
    def __iter__(self):
        return self

    def __next__(self) -> Union[Tuple[Tuple[np.ndarray, torch.FloatTensor], Tuple[np.ndarray, torch.FloatTensor]],
                                Tuple[np.ndarray, np.ndarray]]:
        if self.pos >= len(self.src):
            self.pos = 0
            raise StopIteration
        else:
            batch = (self.src[self.pos:self.pos+self.batch_size], self.tgt[self.pos:self.pos+self.batch_size])
            self.pos += self.batch_size

            return batch
    
    def add_dummy_pairs(self, extra_precursors_path):
        extra_dummy_pairs = []
        with open(extra_precursors_path, 'r') as rf:
            for line in rf:
                extra_dummy_pairs.append(line.strip() + ' ' + line.strip()) # dummies just translate back to same mol
        extra_data = PairDataset(path=extra_dummy_pairs, 
                                 i2s=self.i2s, 
                                 batch_size=self.batch_size)
        extra_data.src_props = np.zeros(len(extra_dummy_pairs))
        extra_data.tgt_props = np.zeros(len(extra_dummy_pairs))
        self.add(extra_data)

    def filter_dummy_pairs(self, need_props=True):
        keep_idx = np.array([i for i in range(len(self.src)) if self.src_smiles[i] != self.tgt_smiles[i]]) # dummies just translate back to same mol
        self.src = self.src[keep_idx]
        self.src_smiles = [self.src_smiles[i] for i in keep_idx]
        if hasattr(self, 'src_props') and need_props:
            self.src_props = self.src_props[keep_idx]
        self.tgt = self.tgt[keep_idx]
        self.tgt_smiles = [self.tgt_smiles[i] for i in keep_idx]
        if hasattr(self, 'tgt_props') and need_props:
            self.tgt_props = self.tgt_props[keep_idx]
    
    def add(self, other):
        self.src = np.concatenate([self.src, other.src], axis=0)
        self.src_smiles += other.src_smiles
        if hasattr(self, 'src_props'):
            self.src_props = np.concatenate([self.src_props, other.src_props], axis=0)
        self.tgt = np.concatenate([self.tgt, other.tgt], axis=0)
        self.tgt_smiles += other.tgt_smiles
        if hasattr(self, 'tgt_props'):
            self.tgt_props = np.concatenate([self.tgt_props, other.tgt_props], axis=0)

    def reshuffle(self, seed: int = 0, need_props: bool = False):
        np.random.seed(seed)
        idx = np.arange(len(self.src))
        np.random.shuffle(idx)
        self.src = self.src[idx]
        self.src_smiles = [self.src_smiles[i] for i in idx]
        self.tgt = self.tgt[idx]
        self.tgt_smiles = [self.tgt_smiles[i] for i in idx]
        if need_props and hasattr(self, 'tgt_props'): # if need_props is false, then the props aren't going to be used further
            self.tgt_props = self.tgt_props[idx]
        if need_props and hasattr(self, 'src_props'):
            self.src_props, self.src_props[idx]
        self.pos = 0
    
    def split(self, fractions: List[float], seed: int = 0) -> List['PairDataset']:
        assert sum(fractions) == 1

        self.reshuffle(seed)
        splits = []
        total_frac = 0
        for i in range(len(fractions)):
            new_dataset = deepcopy(self)
            new_dataset.src = self.src[int(total_frac * len(self.src)):int((total_frac + fractions[i]) * len(self.src))]
            new_dataset.tgt = self.tgt[int(total_frac * len(self.tgt)):int((total_frac + fractions[i]) * len(self.tgt))]
            new_dataset.src_smiles = self.src_smiles[int(total_frac * len(self.src_smiles)):int((total_frac + fractions[i]) * len(self.src_smiles))]
            new_dataset.tgt_smiles = self.tgt_smiles[int(total_frac * len(self.tgt_smiles)):int((total_frac + fractions[i]) * len(self.tgt_smiles))]
            if hasattr(self, 'src_props'):
                new_dataset.src_props = self.src_props[int(total_frac * len(self.src_props)):int((total_frac + fractions[i]) * len(self.src_props))]
            if hasattr(self, 'tgt_props'):
                new_dataset.tgt_props = self.tgt_props[int(total_frac * len(self.tgt_props)):int((total_frac + fractions[i]) * len(self.tgt_props))]
            new_dataset.pos = 0
            splits.append(new_dataset)
            total_frac += fractions[i]

        return splits
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as wf:
            for s, t in zip(self.src_smiles, self.tgt_smiles):
                wf.write(s + ' ' + t + '\n')


class SourceDataset(Dataset):
    def __init__(self,
                 path: str,
                 i2s: List[str],
                 s2i: Dict[str, int],
                 pad_index: int = 0,
                 start_index: int = 1,
                 end_index: int = 2,
                 batch_size: int = 32,
                 max_length: int = 100):
        self.src = []
        self.src_smiles = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                assert ' ' not in line
                self.src.append(list(line[:max_length]))
                self.src_smiles.append(line[:max_length])

        self.i2s = i2s
        self.np_i2s = np.array(self.i2s)
        self.s2i = s2i
        self.pad_index = pad_index
        self.start_index = start_index
        self.end_index = end_index

        for i in range(len(self.src)):
            self.src[i] = [self.start_index] + [self.s2i[char] for char in self.src[i]] + [self.end_index]

        self.src = np.array(self.src)
        self.pos = 0
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.src)
    
    def __iter__(self):
        return self

    def __next__(self) -> Union[Tuple[np.ndarray, np.ndarray],
                                np.ndarray]:
        if self.pos >= len(self.src):
            self.pos = 0
            raise StopIteration
        else:
            batch = self.src[self.pos:self.pos+self.batch_size]
            self.pos += self.batch_size

            return batch
    
    def add(self, other):
        self.src = np.concatenate([self.src, other.src], axis=0)
        self.src_smiles = np.concatenate([self.src_smiles, other.src_smiles], axis=0)
        if hasattr(self, 'src_props'):
            self.src_props = np.concatenate([self.src_props, other.src_props], axis=0)

    def reshuffle(self, seed: int = 0):
        np.random.seed(seed)
        idx = np.arange(len(self.src))
        np.random.shuffle(idx)
        self.src = self.src[idx]
        self.src_smiles = [self.src_smiles[i] for i in idx]
        if hasattr(self, 'src_props'):
            self.src_props, self.src_props[idx]
        self.pos = 0
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as wf:
            for s in self.src_smiles:
                wf.write(s + '\n')
