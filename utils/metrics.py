import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn import metrics

from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix

def roc(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        metric = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
            score = auc(fpr, tpr)
            metric[i]= score
            curves.append((fpr, tpr))
    return metric, curves
    
def pr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves

def tfnp(label, prediction):
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn =0,0,0,0
    
    return tp, tn, fp, fn

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

def accuracy(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(accuracy_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = accuracy_score(label[:,i], np.round(prediction[:,i]))
    return metric


def calculate_metrics(label, prediction, objective):
    """calculate metrics for classification"""
    # import pdb; pdb.set_trace()
    

    if (objective == "binary") | (objective == 'hinge'):
        ndim = np.ndim(label)
        #if ndim == 1:
        #    label = one_hot_labels(label)
        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        # import pdb; pdb.set_trace()
        if ndim == 2:
            prediction=prediction[:,0]
            label = label[:,0]
        # pred_class = prediction[:,0]>0.5
        pred_class = prediction>0.5
        # tp, tn, fp, fn = tfnp(label[:,0], pred_class)
        tp, tn, fp, fn = tfnp(label, pred_class)
        # tn8, fp8, fn8, tp8 = tfnp(label[:,0], prediction[prediction>0.8][:,0])
        # import pdb; pdb.set_trace()
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    elif objective == "categorical":

        correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        for i in range(label.shape[1]):
            label_c, prediction_c = label[:,i], prediction[:,i]
            auc_roc, roc_curves = roc(label_c, prediction_c)
            mean.append(np.nanmean(auc_roc))
            std.append(np.nanstd(auc_roc))


    elif (objective == 'squared_error') | (objective == 'kl_divergence') | (objective == 'cdf'):
        ndim = np.ndim(label)
        #if ndim == 1:
        #    label = one_hot_labels(label)
        label[label<0.5] = 0
        label[label>=0.5] = 1
        # import pdb; pdb.set_trace()

        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        # import pdb; pdb.set_trace()
        if ndim == 2:
            prediction=prediction[:,0]
            label = label[:,0]
        # pred_class = prediction[:,0]>0.5
        pred_class = prediction>0.5
        # tp, tn, fp, fn = tfnp(label[:,0], pred_class)
        tp, tn, fp, fn = tfnp(label, pred_class)
        # mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn]
        # std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        

        # squared_error
        corr = pearsonr(label,prediction)
        rsqr, slope = rsquare(label, prediction)
        # mean = [np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        # std = [np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn, np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

    else:
        mean = 0
        std = 0

    return [mean, std]


class MLMetrics(object):
    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst)>0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg() 

    def compute_avg(self):
        if len(self.metrics)>1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.tp  = int(self.sum[3])
        self.tn  = int(self.sum[4])
        self.fp  = int(self.sum[5])
        self.fn  = int(self.sum[6])
        if len(self.avg)>7:
            self.other = self.avg[7:]