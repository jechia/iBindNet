import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from model.model import build_model
from model.train import train_BERT, validate, compute_gradients, compute_gradients, FineTuneModel, inference, inference_trt
 
from utils.utils import make_directory, param_num, fix_seed, log_print, save_gradients
from utils.utils import load_data, split_dataset, MyDataset, SeqDataset, save_evals, save_infers
from model.tokenizer import Tokenizer
from model.optimizer import AdamW, get_linear_schedule_with_warmup
from model.loss import FocalLoss, GHMC_loss, BCELoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
from utils.utils import get_engine
import tensorrt as trt

#python tools/main.py --train 
#                     --lr 0.0001 
#                     --data /home/huyue/bert_pretrain.txt 
#                     --vocab /home/huyue/6mer.txt 
#                     --p_name pretrain_6mer
#                     --out_dir /home/huyue/program/iBindNet_bert 
#                     --nepochs 20
#                     --workers 4 
#                     --kmer_size 6 
#                     --batch_size 1024
#                     --vocab_size 4101

def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='Official version of PrismNet')
    # Data options
    parser.add_argument('--data_dir',       type=str, default="data_dir", help='data path')
    parser.add_argument('--prefix',       type=str, default="prefix", help='prefix')
    parser.add_argument('--data',       type=str, default="data", help='data')
    parser.add_argument('--vocab',       type=str, default="vocab", help='vocab path')
    parser.add_argument('--p_name',         type=str, default="test", metavar='N', help='protein name')
    parser.add_argument('--out_dir',        type=str, default=".", help='output directory')
    parser.add_argument('--mode',           type=str, default="BERT", help='training mode of network')
    parser.add_argument("--out_file",     type=str, help="out file", default="")
    # Training Hyper-parameter
    parser.add_argument('--arch',           default="BERT", help='network architecture')
    parser.add_argument('--lr',             type=float, default=0.0001, help='learning rate')
    parser.add_argument('--load_best',      action='store_true', help='load best model')
    parser.add_argument('--batch_size',     type=int, default=64, help='input batch size')
    parser.add_argument('--nepochs',        type=int, default=10, help='number of epochs to train')
    parser.add_argument('--kmer_size',    type=int, default=6, help='kmer size')
    parser.add_argument('--hidden_size',    type=int, default=128, help='hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=2, help='number of attention heads')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='the dropout rate')
    parser.add_argument('--intermediate_size', type=int, default=128, help='intermediate size')
    parser.add_argument('--hidden_act', type=str, default='gelu_new', help='hidden activation function')
    parser.add_argument('--max_position', type=int, default=100, help='max position of sequence')
    parser.add_argument('--segment_vocab_size', type=int, default=2, help='segment vocab size')
    parser.add_argument('--vocab_size', type=int, default=4101, help='vocab size')
    parser.add_argument('--early_stopping', type=int, default=100, help='early stopping')
    parser.add_argument('--loss_type', type=str, default='ce', help='the type of loss')
    parser.add_argument('--pos_weight', type=int, default=2, help='weight of the positive class')
    # Training 
    parser.add_argument('--train',       action='store_true', help='pre-train mode')
    parser.add_argument('--multi',       action='store_true', help='single node multiple gpu')
    parser.add_argument('--onnx',       action='store_true', help='convert to onnx model')
    parser.add_argument('--trt',       action='store_true', help='convert to trt engine')
    parser.add_argument('--eval',       action='store_true', help='eval mode')
    parser.add_argument('--finetune',       action='store_true', help='finetune mode')
    parser.add_argument('--infer',          action='store_true', help='infer mode')
    parser.add_argument('--grad',           action='store_true', help='gradient mode')
    parser.add_argument('--model_name',     type=str, help='the model path')
    # misc
    parser.add_argument('--tfboard',        action='store_true', help='tf board')
    parser.add_argument('--no-cuda',        action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed',           type=int, default=1024, help='manual seed')
    parser.add_argument('--local_rank',     type=int, default=0, help='local rank')
    parser.add_argument('--workers',        type=int, help='number of data loading workers', default=4)
    args = parser.parse_args()
    print(args)
    if args.multi:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend='nccl')
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    

    # out dir
    identity   = args.p_name
    make_directory(args.out_dir,"out/")
    model_dir  = make_directory(args.out_dir,"out/models")
    model_path = os.path.join(model_dir, identity+".pth")

    if args.tfboard:
        tfb_dir  = make_directory(args.out_dir,"out/tfb")
        writer = SummaryWriter(tfb_dir)
    else:
        writer = None
    # fix random seed
    fix_seed(args.seed)

    if args.multi:
        device = torch.device("cuda", local_rank)
        print(f"[init] == local_rank: {local_rank}, global rank: {rank} ==")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    
    model = build_model(args.hidden_size, args.num_attention_heads, args.num_hidden_layers, args.dropout_rate, args.intermediate_size, args.hidden_act, args.vocab_size, args.max_position, args.segment_vocab_size)
    param_num(model)
    # print(model)

    pretrain = False
    if args.finetune:
        pretrained_model = model
        filename = os.path.join(model_dir, args.model_name+".pth")
        print("Loading model: {}".format(filename))
        pretrained_model.load_state_dict(torch.load(filename,map_location='cpu'))
        model = FineTuneModel(pretrained_model, args.hidden_size, num_classes=1)
        pretrain=True

    if args.load_best:
        filename = os.path.join(model_dir, args.model_name+".pth")
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename,map_location='cpu'))

    model = model.to(device)   

    print("Network Arch:", args.arch)
    if args.loss_type == "ce":
        criterion = BCELoss(pos_weight=args.pos_weight)
        loss_function = "Binary Cross Entropy"
    elif args.loss_type == "focal":
        criterion = FocalLoss()
        loss_function = "Focal Loss"
    elif args.loss_type == "ghm":
        criterion = GHMC_loss()
        loss_function = "Gradient Harmonizing Mechanism Classification Loss"
    else:
        print("wrong type of loss function!")
        sys.exit()

    tokenizer = Tokenizer(args.vocab,split=args.kmer_size)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

    if args.train:

        print("Using loss function: " + loss_function)

        if args.multi:

            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model,device_ids=[local_rank],output_device=local_rank)
            
            train_fn = args.data_dir + "/" + args.prefix + "_train.txt"
            X_train, y_train = load_data(train_fn)
    
            test_fn = args.data_dir + "/" + args.prefix + "_test.txt"
            X_test, y_test = load_data(test_fn)
    
            batch_size = args.batch_size // torch.cuda.device_count()
            
            train_dataset = MyDataset(X_train, y_train, max_len = args.max_position, tokenizer=tokenizer)
            test_dataset = MyDataset(X_test, y_test, max_len = args.max_position,tokenizer=tokenizer)
    
            train_sampler = DistributedSampler(train_dataset,shuffle=True)
            test_sampler = DistributedSampler(test_dataset,shuffle=True)
    
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,**kwargs)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,**kwargs)
            print("Train set:", len(train_dataloader.dataset))
            print("Test  set:", len(test_dataloader.dataset))
    
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'layerNorm']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
            num_training_steps = (len(train_dataloader) + 1) * args.nepochs
            num_warmup_steps = num_training_steps * 0.05
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            total_step = len(train_dataloader)
    
            best_auc = 0
            best_acc = 0
            best_epoch = 0
            for epoch in range(1, args.nepochs + 1):
    
                train_dataloader.sampler.set_epoch(epoch)
                test_dataloader.sampler.set_epoch(epoch)
    
                t_met =train_BERT(model,device,train_dataloader,criterion,optimizer, scheduler)
                v_met,_,_ = validate(model, device, test_dataloader, criterion)
    
                lr = scheduler.get_lr()[0]
                color_best='green'
                if rank == 0:
                    if best_auc < v_met.auc:
                        best_auc = v_met.auc
                        best_acc = v_met.acc
                        best_epoch = epoch
                        color_best = 'red'
                        filename = model_path
                        if pretrain:
                            torch.save(model.module.pretrained_model.state_dict(), filename)
                        else:
                            torch.save(model.module.state_dict(), filename)
                    if epoch - best_epoch > args.early_stopping:
                        print("Early stop at %d, %s "%(epoch, args.p_name))
                        break
        
                    line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}, AUC: {:.4f} lr: {:.6f}'.format(\
                        args.p_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
                    log_print(line, color='green', attrs=['bold'])
                    
                    line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}, AUC: {:.4f} ({:.4f})'.format(\
                        args.p_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc)
                    log_print(line, color=color_best, attrs=['bold'])
                
            if rank == 0:
                print("{} auc: {:.4f} acc: {:.4f}".format(args.p_name, best_auc, best_acc))

        else:
            
            train_fn = args.data_dir + "/" + args.prefix + "_train.txt"
            X_train, y_train = load_data(train_fn)
    
            test_fn = args.data_dir + "/" + args.prefix + "_test.txt"
            X_test, y_test = load_data(test_fn)

            batch_size = args.batch_size
            
            train_dataset = MyDataset(X_train, y_train, max_len = args.max_position, tokenizer=tokenizer)
            test_dataset = MyDataset(X_test, y_test, max_len = args.max_position,tokenizer=tokenizer)
    
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, **kwargs)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
            print("Train set:", len(train_dataloader.dataset))
            print("Test  set:", len(test_dataloader.dataset))
    
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'layerNorm']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
            num_training_steps = (len(train_dataloader) + 1) * args.nepochs
            num_warmup_steps = num_training_steps * 0.05
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            total_step = len(train_dataloader)
    
            best_auc = 0
            best_acc = 0
            best_epoch = 0
            for epoch in range(1, args.nepochs + 1):
    
                t_met =train_BERT(model,device,train_dataloader,criterion,optimizer, scheduler)
                v_met,_,_ = validate(model, device, test_dataloader, criterion)
    
                lr = scheduler.get_lr()[0]
                color_best='green'
                if best_auc < v_met.auc:
                    best_auc = v_met.auc
                    best_acc = v_met.acc
                    best_epoch = epoch
                    color_best = 'red'
                    filename = model_path
                    if pretrain:
                        torch.save(model.pretrained_model.state_dict(), filename)
                    else:
                        torch.save(model.state_dict(), filename)
                if epoch - best_epoch > args.early_stopping:
                    print("Early stop at %d, %s "%(epoch, args.p_name))
                    break
    
                line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}, AUC: {:.4f} lr: {:.6f}'.format(\
                    args.p_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
                log_print(line, color='green', attrs=['bold'])
                
                line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}, AUC: {:.4f} ({:.4f})'.format(\
                    args.p_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc)
                log_print(line, color=color_best, attrs=['bold'])
                
            print("{} auc: {:.4f} acc: {:.4f}".format(args.p_name, best_auc, best_acc))


    if args.eval:

        test_fn = args.data_dir + "/" + args.prefix + "_test.txt"
        X_test, y_test = load_data(test_fn)

        test_dataset = MyDataset(X_test, y_test, max_len = args.max_position,tokenizer=tokenizer)
        test_sampler = DistributedSampler(test_dataset,shuffle=False)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,**kwargs)
        print("Test  set:", len(test_dataloader.dataset))

        met, y_all, p_all = validate(model, device, test_dataloader, criterion)
        print("> eval {} auc: {:.4f} acc: {:.4f}".format(args.p_name, met.auc, met.acc))
        save_evals(args.out_dir, identity, p_all, y_all, met)

    if args.onnx:

        sequences,targets = load_data(args.data, infer=True)

        test_dataset = MyDataset(sequences, targets, max_len = args.max_position,tokenizer=tokenizer)
        data = test_dataset[0]
        inputs = {
                  'token_ids':      data[0].to(device).reshape(1, args.max_position),
                  'segment_ids':    data[1].to(device).reshape(1, args.max_position)
                 }
        
        print(inputs['token_ids'].dtype)
        
        model.eval()
        
        outdir = make_directory(args.out_dir, "out/onnxs")
        onnx_path = os.path.join(outdir, args.model_name+'.onnx')
        print(onnx_path)
        
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}

        torch.onnx.export(model,
                  args = tuple(inputs.values()),
                  f=onnx_path,
                  do_constant_folding=True,
                  input_names=['tokens_ids', 'segment_ids'],
                  output_names=['output'],
                  dynamic_axes={'tokens_ids': symbolic_names,
                                'segment_ids' : symbolic_names,
                                'output' : {0:'batch_size'}}
                  )

    if args.trt:
        
        outdir = os.path.join(args.out_dir, "out/trts")
        trt_path = os.path.join(outdir, args.model_name+'.trt')
        
        engine = get_engine(trt_path)
        sequences,targets = load_data(args.data, infer=True)

        test_dataset = MyDataset(sequences, targets, max_len = args.max_position,tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,**kwargs)
        print("Test  set:", len(test_dataloader.dataset))
        
        p_all = inference_trt(engine, test_dataloader)
        
        save_infers(args.out_dir, identity, p_all)

    if args.infer:

        if args.multi:

            model = DDP(model,device_ids=[local_rank],output_device=local_rank)

            batch_size = args.batch_size // torch.cuda.device_count()
            sequences,targets = load_data(args.data, infer=True)
            test_dataset = MyDataset(sequences, targets, max_len = args.max_position,tokenizer=tokenizer)
            
            test_sampler = DistributedSampler(test_dataset,shuffle=False)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,**kwargs)
            print("Test  set:", len(test_dataloader.dataset))

            p_all = inference(model, device, test_dataloader)
            save_infers(args.out_dir, identity, p_all)

        else:
            sequences,targets = load_data(args.data, infer=True)

            test_dataset = MyDataset(sequences, targets, max_len = args.max_position,tokenizer=tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,**kwargs)
            print("Test  set:", len(test_dataloader.dataset))

            p_all = inference(model, device, test_dataloader)
            save_infers(args.out_dir, identity, p_all)

    if args.grad:
    
        sequences,targets = load_data(args.data)
        test_dataset = SeqDataset(sequences, targets)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        gradients,preds,labels = compute_gradients(model, tokenizer, test_dataloader, device,custom_labels=[0,1],pretrain =False,batch_size=args.batch_size)
        identity = args.out_file
        grad_dir  = make_directory(args.out_dir,"out/grads")
        save_gradients(grad_dir, identity, gradients,preds,labels)
    
if __name__ == '__main__':
    main()
