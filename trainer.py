"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

import re
import pandas as pd


logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e2 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e7 # (at what point we reach 10% of original LR)
    cdq = False
    multi_model = False
    # checkpoint settings
    ckpt_path = None
    run_name = None
    num_workers = 0 # for DataLoader
    load_checkpoint_path = None

    max_iter = 50000

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, config):   # train_dataset, valid_dataset, test_dataset,
        self.model = model
        self.train_dataset = None  # train_dataset
        self.test_dataset = None  # test_dataset
        self.valid_dataset = None  # valid_dataset
        self.config = config
        self.tokens = 0

        # take over whatever gpus are on the system
        self.device = 'cpu'
        print('dist:', config.dist)
        if config.dist:
            self.device = config.rank
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])
        elif torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, model, best_loss, optimizer, tokens, scaler, save_path):
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Include scaler state
            'tokens': tokens,
            'best_loss': best_loss,
        }
        if self.config.dist:
            if self.device == 0:
                torch.save(checkpoint, save_path)
        else:
            torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path, optimizer, scaler):
        checkpoint = torch.load(load_path, map_location='cuda')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokens = checkpoint['tokens']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']

    def train(self, wandb):
        # if self.config.dist:
        #     if self.device == 0:
        #         wandb.init(
        #             entity='grelu',
        #             project="RNA-optimization",
        #             job_type='FA',
        #             name='train_Enformer'
        #         )
        # else:
        #     wandb.init(
        #         entity='grelu',
        #         project="RNA-optimization",
        #         job_type='FA',
        #         name='train_Enformer'
        #     )
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        # optimizer = torch.optim.AdamW(raw_model.parameters(), lr=config.learning_rate, betas=config.betas)
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        if config.load_checkpoint_path is not None:
            print(f'resuming training from {config.load_checkpoint_path}...')
            start_epoch, best_loss = self.load_checkpoint(config.load_checkpoint_path, optimizer, scaler)
            # model = self.model
        else:
            start_epoch = -1
            best_loss = float('inf')
            self.tokens = 0  # counter used for learning rate decay

        # if self.config.dist:
        #     sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=False)
        #     loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
        #                         batch_size=config.batch_size,
        #                         sampler=sampler)
        # else:
        #     sampler = None
        #     # loader = None
        #     loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
        #                         batch_size=config.batch_size,
        #                         num_workers=config.num_workers)

        if self.config.dist:
            if self.device == 0:
                # wandb.define_metric("val_MSE", step_metric="val_step")
                wandb.define_metric("test_MSE", step_metric="test_step")
                wandb.define_metric("train_MSE", step_metric="train_step")
        else:
            # wandb.define_metric("val_MSE", step_metric="val_step")
            wandb.define_metric("test_MSE", step_metric="test_step")
            wandb.define_metric("test_pearsonR", step_metric="test_step")
            wandb.define_metric("train_MSE", step_metric="train_step")
        def run_epoch(split, epoch):
            is_train = (split == 'train')
            model.train(is_train)
            # if is_train:
            #     if self.config.dist:
            #         sampler.set_epoch(epoch)
                # else:
                #     loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                #                         batch_size=config.batch_size,
                #                         num_workers=config.num_workers)
            # if not is_train:
            #     if split == 'val':
            #         data = self.valid_dataset
            #     else:
            #         data = self.test_dataset
            #     # self.save_checkpoint()
            #     # if self.config.dist:
            #     #     sampler = torch.utils.data.distributed.DistributedSampler(data)
            #     #     loader = DataLoader(data, shuffle=False, pin_memory=True,
            #     #                         batch_size=config.batch_size,
            #     #                         sampler=sampler)
            #     # else:
            #     loader = DataLoader(data, shuffle=False, pin_memory=True,
            #                         batch_size=config.batch_size,
            #                         num_workers=config.num_workers)

            losses = []
            # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # for it, (input_ids, labels, texts, EOS) in pbar:  #input_ids, attention_masks, labels
            for it in range(config.max_iter):


                # place data on the correct device
                # input_ids = input_ids.to(self.device)
                # # attention_masks = attention_masks.to(self.device)
                # labels = labels.to(self.device)
                # EOS = EOS.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        # loss, logits, _, _ = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)   #bert
                        # loss = model(input_ids, labels, texts, EOS)  #Enformer
                        if config.multi_model:
                            loss, multimodel_losses = model()
                        else:
                            loss = model()
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (input_ids >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += 32*128*200*4
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if (it + epoch*config.max_iter) % 500 == 0:
                        print(f"step_train_loss: {loss} train_step: {it + epoch*config.max_iter}, learning_rate: {lr}")
                    if self.config.dist:
                        if self.device == 0:
                            wandb.log({'train_MSE': loss, 'train_step': int(it + epoch * config.max_iter)})
                        if (it + epoch * config.max_iter) % 2000 == 0:  # it != 0 and
                            val_loss = eval_seq_step('val', epoch, ((it + epoch * config.max_iter) / 2000))
                            test_loss = eval_seq_step('test', epoch, ((it + epoch * config.max_iter) / 2000))
                            print(f"step: {it + epoch * config.max_iter}, val_loss: {val_loss}, test_loss: {test_loss}")
                    else:
                        if config.multi_model:
                            for time, loss_step in enumerate(multimodel_losses):
                                wandb.log({'train_MSE': loss_step, 'train_step': int(time + 10*(it + epoch * config.max_iter))})
                        else:
                            wandb.log({'train_MSE': loss, 'train_step': int(it + epoch * config.max_iter)})
                        if (it + epoch * config.max_iter) % 200 == 0:   #it != 0 and
                            # val_loss = eval_seq_step('val', epoch, ((it + epoch * len(loader))/2000))
                            test_loss = eval_seq_step('test', epoch, ((it + epoch * config.max_iter)/200))
                            print(f"step: {it + epoch*config.max_iter}, test_loss: {test_loss}")   # val_loss: {val_loss},
                            ckpt_path = f'../cond_gpt/weights/{self.config.run_name}_it{it + epoch * config.max_iter}.pt'
                            print(f'Saving at latest epoch: {ckpt_path}')
                            self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)
                            model.train(is_train)
                    # pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    
            if is_train:
                return float(np.mean(losses)), test_loss

            if not is_train:
                test_loss = float(np.mean(losses))
                print("eval loss: %f", test_loss)
                return test_loss

        def eval_seq_step(split, epoch, num_run):
            is_train = (split == 'train')
            model.train(is_train)

            # if split == 'val':
            #     data = self.valid_dataset
            # else:
            #     data = self.test_dataset
            # # self.save_checkpoint()
            # print(f"Sample number for eval: {data.sample_num}")
            # if self.config.dist:
            #     sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False)
            #     loader = DataLoader(data, shuffle=False, pin_memory=True,
            #                         batch_size=data.sample_num,
            #                         sampler=sampler)
            # else:
            #     loader = DataLoader(data, shuffle=False, pin_memory=True,
            #                         batch_size=data.sample_num,
            #                         num_workers=config.num_workers)

            # losses = []
            # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # for it, (input_ids, labels) in pbar:  # input_ids, attention_masks, labels

                # # place data on the correct device
                # input_ids = input_ids.to(self.device)
                # # attention_masks = attention_masks.to(self.device)
                # labels = labels.to(self.device)

            # forward the model
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(is_train):
                    # loss, logits, _, _ = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)   #bert
                    losses, pearsons = model.module.evaluate_seq_step()  # Enformer
                    # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    # losses.append(loss.item())
                # if self.config.dist:
                #     if self.device == 0:
                #         if split == 'val':
                #             wandb.log({'val_MSE': loss, 'val_step': int(it + num_run * 200)})
                #         else:
                #             wandb.log({'test_MSE': loss, 'test_step': int(it + num_run * 200)})
                # else:
                #     if split == 'val':
                #         wandb.log({'val_MSE': loss, 'val_step': int(it+num_run*200)})
                #     else:
            for it, loss in enumerate(losses):
                wandb.log({'test_MSE': loss, 'test_step': int(it+num_run*128)})
            for it, pearson in enumerate(pearsons):
                wandb.log({'test_pearsonR': pearson, 'test_step': int(it+num_run*128)})
            print(f"last eval step: {127+num_run*128}")

            test_loss = np.mean(losses)
            test_pearson = np.mean(pearsons)
            print("eval loss: %f", test_loss, "pearsonR: %f", test_pearson)
            return test_loss


        for epoch in range(start_epoch+1, config.max_epochs):

            train_loss, test_loss = run_epoch('train', epoch)
            # if self.test_dataset is not None:
            #     test_loss = eval_seq_step('test', epoch)
            # print(f"epoch_valid_loss: {test_loss}, epoch_train_loss: {train_loss}, epoch: {epoch + 1}")
            print(f"epoch_train_loss: {train_loss}, epoch: {epoch + 1}")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}: {self.config.ckpt_path}')
                # self.save_checkpoint()
                self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, self.config.ckpt_path)

            if ((epoch+1) >= self.config.save_start_epoch and (epoch+1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                # last_model = self.model.module if hasattr(self.model, "module") else self.model
                ckpt_path = f'../cond_gpt/weights/{self.config.run_name}_ep{epoch+1}.pt'
                print(f'Saving at latest epoch {epoch + 1}: {ckpt_path}')
                if self.config.dist:
                    if self.device == 0:
                        self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)
                        # torch.save(last_model.state_dict(), ckpt_path)
                else:
                    # logger.info("saving %s", ckpt_path)
                    # torch.save(last_model.state_dict(), ckpt_path)
                    self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)



        return None
