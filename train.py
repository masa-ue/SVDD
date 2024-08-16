import pandas as pd
import random
import argparse
import numpy as np
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




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

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


def run_DDP(rank, world_size, args):
    setup(rank, world_size)
    run(args, rank)
    cleanup()


def run(args, rank=None):
    set_seed(args.seed)
    args_dict = vars(args)
    wandb.init(
        entity='grelu',
        project="RNA-optimization",
        job_type='FA',
        name=args.run_name,
        # track hyperparameters and run metadata
        config=args_dict
    )
    # os.environ["WANDB_MODE"] = "dryrun"

    print("making tokenizer")
    # tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=FLAGS.max_tokenizer_len)
    # print("tokenizer:", tokenizer)
    max_len = args.max_len
    print("tokenizer:")
    tokenizer_path = args.output_tokenizer_dir
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    tokenizer_path = args.output_tokenizer_dir + "/vocab.json"
    print(tokenizer_path)
    if os.path.exists(tokenizer_path):
        print(f"The file '{tokenizer_path}' exists")
        tokenizer = load_tokenizer(tokenizer_path, max_len)
    else:
        tokenizer = SimpleDNATokenizer(max_length=max_len)
        if args.pre_root_path is not None:
            tokenizer.fit_on_file(args.pre_root_path + '.txt')
            tokenizer.fit_on_file(args.pre_root_path + '_val.txt')
        # tokenizer.fit_on_file(args.pre_root_path + '_test.txt')
        tokenizer.fit_on_file(args.root_path + '.txt')
        tokenizer.fit_on_file(args.root_path + '_val.txt')
        if args.conditions_path is not None:
            tokenizer.fit_on_file(args.conditions_path + '.txt')
            tokenizer.fit_on_file(args.conditions_path + '_val.txt')
        tokenizer.save_vocab(tokenizer_path)
        print("tokenizer saved")

    print(tokenizer.get_vocab())  # Print vocabulary
    vocab_size = tokenizer.get_vocab_size()

    print("making dataset")
    # with open(args.root_path + '.txt', 'r') as file:
    #     train_data = file.readlines()
    # file.close()
    # # train_data = load_dataset_from_files(args.root_path, 'train', 56)
    # with open(args.root_path + '_val.txt', 'r') as file:
    #     val_data = file.readlines()
    # file.close()
    # # val_data = load_dataset_from_files(args.root_path, 'valid', 7)
    # if args.conditions_path is not None:
    #     print("loading conditions")
    #     with open(args.conditions_path + '.txt', 'r') as file:
    #         conditions_data = file.readlines()
    #     file.close()
    #     with open(args.conditions_path + '_val.txt', 'r') as file:
    #         conditions_data_val = file.readlines()
    # else:
    #     conditions_data = None
    #     conditions_data_val = None
    # if args.conditions_split_id_path is not None:
    #     print("loading conditions split id")
    #     with open(args.conditions_split_id_path + '.txt', 'r') as file:
    #         conditions_split_id = file.readlines()
    #     file.close()
    #     with open(args.conditions_split_id_path + '_val.txt', 'r') as file:
    #         conditions_split_id_val = file.readlines()
    # else:
    #     conditions_split_id = None
    #     conditions_split_id_val = None
    if args.load_checkpoint_path:
        load_checkpoint_path = args.load_checkpoint_path
    else:
        load_checkpoint_path = None


    # if args.model == 'enformer':
    #     train_dataset = DNA_reg_conv_Dataset(mode='train')
    #     valid_dataset = DNA_reg_conv_Dataset(mode='val')
    #     test_dataset = DNA_reg_conv_Dataset(mode='test')
    # else:
    #     train_dataset = DNA_reg_Dataset(tokenizer, max_len, mode='train')
    #     valid_dataset = DNA_reg_Dataset(tokenizer, max_len, mode='val')
    #     test_dataset = DNA_reg_Dataset(tokenizer, max_len, mode='test')
    # print(f"train dataset size: {len(train_dataset)}")
    # print(f"val dataset size: {len(valid_dataset)}")
    # print(f"test dataset size: {len(test_dataset)}")
    print(f"max iterations: {args.max_iters}")

    # if args.conditions_path is not None or args.conditions_split_id_path is not None or args.fix_condition is not None:
    #     isconditional = True
    # else:
    #     isconditional = False

    print("loading model")
    multi_model = False
    if args.model == 'enformer':
        # common_trunk = EnformerTrunk(n_conv=args.n_conv, channels=args.channels, n_transformers=args.n_transformers,
        #                              n_heads=args.n_heads, key_len=args.key_len,
        #                              attn_dropout=args.attn_dropout, pos_dropout=args.pos_dropout,
        #                              ff_dropout=args.ff_dropout, crop_len=args.crop_len)
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=args.n_task, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size, val_batch_num=args.val_batch_num, task=args.task, n_tasks=args.n_task)
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
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('total params:', sum(p.numel() for p in model.parameters()))
    os.makedirs(f'../cond_gpt/weights/', exist_ok=True)
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          multi_model=multi_model, lr_decay=True, warmup_tokens=12800*args.batch_size * max_len * 4,
                          final_tokens=128 * args.max_iters * args.batch_size * max_len * 4, max_iter=args.max_iters,
                          num_workers=args.num_workers, ckpt_path=f'../cond_gpt/weights/{args.run_name}.pt',
                          run_name=args.run_name, block_size=max_len, generate=False, save_start_epoch=args.save_start_epoch,
                          grad_norm_clip=args.grad_norm_clip, load_checkpoint_path=load_checkpoint_path,
                          save_interval_epoch=args.save_interval_epoch, cdq=args.cdq, dist=args.dist, rank=rank)
    trainer = Trainer(model, tconf)  # train_dataset, valid_dataset, test_dataset,
    df = trainer.train(wandb)
    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--task', type=str, default="rna_saluki",
                        help="task", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
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
    parser.add_argument('--n_task', type=int, default=1,
                        help="number of task head", required=False)
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)
    parser.add_argument('--val_batch_num', type=int, default=64,
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
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
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

    if args.dist:
        world_size = torch.cuda.device_count()
        mp.spawn(run_DDP,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        run(args)
