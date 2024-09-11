import pandas as pd
import random
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
import math
import re
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import RobertaConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_absolute_error
from functools import partial

# from roberta_regression import RobertaForRegression, BertForSequenceClassification
from trainer import Trainer, TrainerConfig
from dataset import DNA_reg_Dataset, SimpleDNATokenizer, DNA_reg_conv_Dataset
from Enformer import BaseModel, BaseModelMultiSep, ConvHead, EnformerTrunk, TimedEnformerTrunk

import wandb 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_from_files(root_path, split, ids):
    dataset = []
    for id in range(ids):
        with open(root_path+'_'+split+'_'+str(id)+'.txt', 'r') as file:
            dataset.extend(file.readlines())
            print('loaded dataset from '+root_path+'_'+split+'_'+str(id)+'.txt')
    return dataset

def load_tokenizer(tokenizer_path,max_length):
    tokenizer = SimpleDNATokenizer(max_length)  # Update max_length if needed
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def run(args, rank=None):
    set_seed(args.seed)
    args_dict = vars(args)
    wandb.init(
        project="DNA-optimization",
        job_type='FA',
        name='decode',
        # track hyperparameters and run metadata
        config=args_dict
    )
    # os.environ["WANDB_MODE"] = "dryrun"

    if args.load_checkpoint_path:
        load_checkpoint_path = args.load_checkpoint_path
    else:
        load_checkpoint_path = None


    print("loading model")
    multi_model = False
    if args.model == 'enformer':
        # common_trunk = EnformerTrunk(n_conv=args.n_conv, channels=args.channels, n_transformers=args.n_transformers,
        #                              n_heads=args.n_heads, key_len=args.key_len,
        #                              attn_dropout=args.attn_dropout, pos_dropout=args.pos_dropout,
        #                              ff_dropout=args.ff_dropout, crop_len=args.crop_len)
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size,
                          val_batch_num=1, task=args.task, n_tasks=args.n_task, saluki_body=args.saluki_body)
    elif args.model == 'multienformer':
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModelMultiSep(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size, val_batch_num=args.val_batch_num)
        multi_model = True
    elif args.model == 'timedenformer':
        common_trunk = TimedEnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size,
                          val_batch_num=args.val_batch_num, timed=True)
    else:
        raise NotImplementedError

    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'], strict=True)
    if load_checkpoint_path is not None:
        print("loading stored model: ", load_checkpoint_path)
        checkpoint = torch.load(load_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('total params:', sum(p.numel() for p in model.parameters()))

    model.cuda()
    model.eval()

    gen_samples, value_func_preds, reward_model_preds, selected_baseline_preds, baseline_preds = model.controlled_decode_TDS(gen_batch_num=args.val_batch_num, sample_M=args.sample_M, alpha = args.alpha )

    hepg2_values_ours_value_func = value_func_preds.cpu().numpy()

    hepg2_values_ours = reward_model_preds.cpu().numpy()
    hepg2_values_selected = selected_baseline_preds.cpu().numpy()
    hepg2_values_baseline = baseline_preds.cpu().numpy()
    print(hepg2_values_baseline.shape)
    np.savez( "./log/%s-%s_TDS" %(args.task, args.reward_name), decoding = hepg2_values_ours, baseline = hepg2_values_baseline)


    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--task', type=str, default="rna_saluki",
                        help="task", required=False)
    parser.add_argument('--saluki_body', type=int, default=0,
                        required=False)
    parser.add_argument('--n_task', type=int, default=1,
                        help="number of task head", required=False)
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",
                        required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--model', type=str, default='enformer',
                        help="name of the model", required=False)
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="name of the tokenizer", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=1,
                        help="total epochs", required=False)
    parser.add_argument('--max_iters', type=int, default=50000,
                        help="total iterations", required=False)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size", required=False)
    parser.add_argument('--sample_M', type=int, default=20,
                        help="sample width", required=False)
    parser.add_argument('--val_batch_num', type=int, default=1,
                        help="val batches", required=False)
    parser.add_argument('--num_workers', type=int, default=12,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--save_start_epoch', type=int, default=120,
                        help="save model start epoch", required=False)
    parser.add_argument('--save_interval_epoch', type=int, default=10,
                        help="save model epoch interval", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=2e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--max_len', type=int, default=512,
                        help="max_len", required=False)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="alph", required=False)
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
    parser.add_argument('--reward_name', type=str, default='HepG2',
                        help="Plot Y axis name", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--auto_fp16to32', action='store_true',
                        default=False, help='Auto casting fp16 tensors to fp32 when necessary')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help="Path to load training checkpoint (if resuming training)", required=False)
    parser.add_argument('--pre_root_path', default=None,
                        help="Path to the pretrain data directory", required=False)
    parser.add_argument('--pre_model_path', default=None,
                        help="Path to the pretrain model", required=False)
    parser.add_argument('--root_path', type=str, default='/home/lix361/projects/rna_optimization/generative/5UTR_Ensembl_cond',
                        help="Path to the root data directory", required=False)
    parser.add_argument('--output_tokenizer_dir', type=str,
                        default='/home/lix361/projects/rna_optimization/generative/storage/5UTR_Ensembl_cond_seq/tokenizer',
                        help="Path to the saved tokenizer directory", required=False)
    parser.add_argument('--fix_condition', default=None,
                        help="fixed condition num", required=False)
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False) 
    parser.add_argument('--conditions_split_id_path', default=None,
                        help="Path to the conditions_split_id", required=False)
    parser.add_argument('--cdq', action='store_true',
                        default=False, help='CD-Q')
    parser.add_argument('--dist', action='store_true',
                        default=False, help='use torch.distributed to train the model in parallel')
    args = parser.parse_args()

    run(args)
