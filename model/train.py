from __future__ import print_function
import argparse, os, copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import MLMetrics
import torch.nn.functional as F
from utils.utils import make_directory
from explainer.sequence_classification import SequenceClassificationExplainer
import utils.common as common

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def train_BERT(model, device, train_loader, criterion, optimizer, scheduler):
    model.train()
    met = MLMetrics(objective='binary')
    for batch_idx, (token_ids, segment_ids, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        token_ids = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        labels = labels.to(device)
        outputs = model(token_ids, segment_ids)

        prob = torch.sigmoid(outputs)

        loss = criterion(outputs, labels)

        y_np = labels.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np,[loss.item()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

    return met

def validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (token_ids, segment_ids, labels) in enumerate(test_loader):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)
            outputs = model(token_ids, segment_ids)
            
            prob = torch.sigmoid(outputs)

            loss = criterion(outputs, labels)

            y_np = labels.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)
    
    met = MLMetrics(objective='binary')
    met.update(y_all, p_all,[l_all.mean()])
    
    return met, y_all, p_all

def inference(model, device, test_loader):
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (token_ids, segment_ids, labels) in enumerate(test_loader):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            outputs = model(token_ids, segment_ids)
            
            prob = torch.sigmoid(outputs)

            p_np = prob.to(device='cpu').numpy()

            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    
    return p_all

def inference_trt(engine, test_loader):
    context = engine.create_execution_context()
    p_all = []
    for batch_idx, (token_ids, segment_ids, labels) in enumerate(test_loader):
        token_ids = to_numpy(token_ids.int())
        segment_ids = to_numpy(segment_ids.int())
        context.active_optimization_profile = 0
        origin_inputshape = context.get_binding_shape(0)
        origin_inputshape[0],origin_inputshape[1] = token_ids.shape
        context.set_binding_shape(0, (origin_inputshape))           
        context.set_binding_shape(1, (origin_inputshape))

        inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
        inputs[0].host = token_ids
        inputs[1].host = segment_ids

        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        trt_outputs = trt_outputs[0].reshape(origin_inputshape[0], -1) # (batch_size,15)
        ort_outs_logits = torch.tensor(trt_outputs)
        prob = torch.sigmoid(ort_outs_logits)

        p_np = prob.to(device='cpu').numpy()

        p_all.append(p_np)

    p_all = np.concatenate(p_all)
    
    return p_all


def compute_gradients(model, tokenizer, test_loader, device, custom_labels,pretrain,batch_size):
    gradients = []
    preds = []
    labels =[]
    cls_explainer = SequenceClassificationExplainer(model=model, tokenizer = tokenizer,device=device,                                         custom_labels=custom_labels,pretrain=pretrain)
    
    for batch_idx, (text, label) in enumerate(test_loader):
        word_attributions ,pred= cls_explainer(text,class_name=label,n_steps=50,internal_batch_size=len(text))
        for i in range(len(pred)):
            gradients.append(word_attributions[i])
            preds.append(int(pred[i]))
            labels.append(int(label[i]))
    return gradients,preds,labels


class FineTuneModel(nn.Module):

    def __init__(self, pretrained_model, hidden_size, num_classes):
        super(FineTuneModel, self).__init__()

        self.pretrained_model = pretrained_model

        new_classification_layer = nn.Linear(hidden_size, 1)
        self.pretrained_model.classification_layer = new_classification_layer

    def forward(self, token_ids, segment_ids=None,attention_mask=None):
        classification_outputs = self.pretrained_model(token_ids, segment_ids, attention_mask)
        return classification_outputs
