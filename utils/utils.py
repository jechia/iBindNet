import numpy as np
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Callable, List, Optional, Set, Tuple, Union

import re

import tensorrt as trt

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy??,???padding?????
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)

def truncate_sequences(maxlen, indices, *sequences):
    """?????????maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences

import pandas as pd

def load_data(path,infer=False):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)

    Seq  = 0
    Score = 1
    
    if df[Seq].str.contains('U').any() == True:
        df[Seq].replace('U','T',regex=True, inplace = True)
        
    sequences = df[Seq].to_numpy()
    if infer:
        targets   = np.zeros(len(df[Seq]))
    else:
        targets   = df[Score].to_numpy()
    return sequences, targets

def split_dataset(data, targets, valid_frac=0.2):
    
    ind0 = np.where(targets<0.5)[0]
    ind1 = np.where(targets>=0.5)[0]
    
    n_neg = int(len(ind0)*valid_frac)
    n_pos = int(len(ind1)*valid_frac)

    shuf_neg = np.random.permutation(len(ind0))
    shuf_pos = np.random.permutation(len(ind1))

    X_train = np.concatenate((data[ind1[shuf_pos[n_pos:]]], data[ind0[shuf_neg[n_neg:]]]))
    Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))
    train = (X_train, Y_train)

    X_test = np.concatenate((data[ind1[shuf_pos[:n_pos]]], data[ind0[shuf_neg[:n_neg]]]))
    Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))
    test = (X_test, Y_test)

    return train, test

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y, max_len, tokenizer):
        self.X = X
        self.y = y
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sentence = self.X[index]
        label = self.y[index]
        tokens_ids, segments_ids = self.tokenizer.encode(sentence, max_len=self.max_len)
        tokens_ids = tokens_ids + (self.max_len - len(tokens_ids)) * [0]
        segments_ids = segments_ids + (self.max_len - len(segments_ids)) * [0]
        tokens_ids_tensor = torch.tensor(tokens_ids)
        segment_ids_tensor = torch.tensor(segments_ids)
        return tokens_ids_tensor, segment_ids_tensor, label

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sentence = self.X[index]
        label = self.y[index]
        return sentence, label

def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

def save_seqs(seq,label,fn):
    with open(fn,"w") as fOut:
        for s,l in zip(seq,label):
            fOut.writelines(s+"\t"+str(l)+"\n")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
        

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")

def param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0-num_param1)
    print("===========================")


try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def save_gradients(out_dir, filename,gradients, predictions,labels):
    evals_dir = out_dir
    grads_path = os.path.join(evals_dir, filename+'_gradients.npz')
    np.savez("test.npz",grad=gradients,pred=predictions,label=labels)
    print("Prediction file:", grads_path)

def save_infers(out_dir, filename, predictions):
    evals_dir = make_directory(out_dir, "out/infer")
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:f}".format(predictions[i,0]), file=f)
    print("Prediction file:", probs_path)


def save_evals(out_dir, filename, predictions, label, met):
    evals_dir = make_directory(out_dir, "out/evals")
    metrics_path = os.path.join(evals_dir, filename+'.metrics')
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(metrics_path,"w") as f:
        if "_reg" in filename:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
                met.avg[7],
                met.avg[8],
                met.avg[9],
            ), file=f)
        else:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
            ), file=f)
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:.3f}\t{}".format(predictions[i,0], label[i,0]), file=f)
    print("Evaluation file:", metrics_path)
    print("Prediction file:", probs_path)


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index
