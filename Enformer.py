import numpy as np
import copy
import torch
from torch import Tensor, einsum, nn
from torch.nn import functional as F
from einops import rearrange
from typing import List, Optional, Union
from enformer_pytorch.modeling_enformer import GELU, AttentionPool, relative_shift
from enformer_pytorch.modeling_enformer import Attention, exponential_linspace_int
# from mamba import MambaLMHeadModel, MambaConfig
# from gpt_model import GPT, GPTConfig
# from mamba import MambaLMHeadModel, MambaConfig
from metric import PearsonR
from scipy.stats import pearsonr
import diffusion_gosai
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import dataloader_gosai
from grelu.lightning import LightningModel

class BaseModel(nn.Module):
    """
    Base model class
    """

    def __init__(self, embedding: nn.Module, head: nn.Module, cdq, batch_size, val_batch_num, timed=False, task="rna_saluki", n_tasks=1, saluki_body=0) -> None:
        super().__init__()
        self.task = task
        self.n_tasks = n_tasks
        self.saluki_body = saluki_body
        if self.task == "rna_saluki" or self.task == "rna":
            self.embedding = ConvGRUTrunk(
                stem_in_channels=4,
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            self.head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
        else:
            self.embedding = embedding
            self.head = head
        self.loss_fct = nn.MSELoss()
        if self.task == "dna":
            self.pearsonr = PearsonR(num_targets=1)
        self.cdq = cdq
        self.timed = timed
        # self.mapping = {"A": 0, "C": 1, "G": 2, "T": 3} N: 4
        # self.num_features = len(self.mapping)

        # ref_model_path = "/gstore/data/resbioai/lix361/rna_optimization/storage/human_enhancer_seq_16_16_768_64_ep150_ep122.pt"
        # print(f"ref model: GPT, {ref_model_path}")
        # mconf = GPTConfig(vocab_size, 202, num_props=0, n_layer=16, n_head=16, n_embd=768, scaffold=False,
        #                   scaffold_maxlen=202, lstm=False, lstm_layers=0, isconditional=False)
        # self.ref_model = GPT(mconf)

        # ref_model_path = "/home/lix361/projects/rna_optimization/cond_gpt/weights/human_enhancer_seq_mamba_32_768_64_ep150_ep127.pt"
        # print(f"ref model: Mamba, {ref_model_path}")
        # mamba_config = MambaConfig(d_model=768, n_layer=32, vocab_size=vocab_size,
        #                            num_props=0, scaffold=False, isconditional=False,auto_fp16to32=True)
        # self.ref_model = MambaLMHeadModel(mamba_config)

        self.NUM_SAMPLES_PER_BATCH = batch_size

        if self.task == "rna" or self.task == "rna_saluki":
            CKPT_PATH = 'artifacts/RNA_Diffusion:v0/best.ckpt'
            print("CKPT_PATH: ", CKPT_PATH)
            GlobalHydra.instance().clear()
            # Initialize Hydra and compose the configuration
            initialize(config_path="configs_gosai_rna", job_name="load_model")
            cfg = compose(config_name="config_gosai.yaml")
        else:
            CKPT_PATH = 'artifacts/DNA_Diffusion:v0/last.ckpt'
            print("CKPT_PATH: ", CKPT_PATH)
            # reinitialize Hydra
            GlobalHydra.instance().clear()
            # Initialize Hydra and compose the configuration
            initialize(config_path="configs_gosai", job_name="load_model")
            cfg = compose(config_name="config_gosai.yaml")

        # Initialize the model
        self.ref_model = diffusion_gosai.Diffusion.load_from_checkpoint(CKPT_PATH, config=cfg, map_location='cpu')
        # self.detokenizer = dataloader_gosai.DNASequenceDetokenizer()

        # self.ref_model.load_state_dict(torch.load(ref_model_path, map_location='cpu')['model_state_dict'], strict=True)
        # self.tokenizer = tokenizer
        self.ref_model.cuda()
        self.ref_model.eval()
        # Freeze the ref_model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False

        if self.task == "rna_old":
            self.reward_model = LightningModel.load_from_checkpoint("artifacts/DNA_Diffusion:v0/last.ckpt", map_location='cpu')
        elif self.task == "rna":
            self.reward_model = LightningModel.load_from_checkpoint("artifacts/RNA_evaluation:v0/model.ckpt", map_location='cpu')
        elif self.task == "rna_saluki":
            common_trunk = ConvGRUTrunk(
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
            self.reward_model = OriBaseModel(embedding=common_trunk, head=human_head)
            ckpt_human = torch.load("artifacts/RNA_Stability_oracle:v0/rna_saluki_diffusion_enformer_7_11_1536_16_ep10_it3200.pt", map_location='cpu')
            self.reward_model.load_state_dict(ckpt_human, strict=True)
        else:
            self.reward_model = LightningModel.load_from_checkpoint("artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')
        self.reward_model.cuda()
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.val_data_num = val_batch_num * batch_size
        # Initialize lists to store all one-hot samples and targets by time steps across all validation batches
        all_time_step_samples = [[] for _ in range(128)]
        all_time_step_targets = [[] for _ in range(128)]
        for i in range(val_batch_num):
            samples, mid_samples = self.ref_model._sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(samples)
            if self.task == "rna_saluki":
                target = self.reward_model(self.transform_samples_saluki(samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach() 
            onehot_mid_samples = [self.transform_samples(sample) for sample in mid_samples]
            onehot_mid_samples.append(onehot_samples)
            # targets = [target for _ in range(len(onehot_mid_samples))]
            # x0 = torch.cat(onehot_mid_samples, dim=0)
            # x0 = x0.float()
            # y = torch.cat(targets, dim=0)
            # Store samples and targets in corresponding time step lists
            for j, sample in enumerate(onehot_mid_samples):
                all_time_step_samples[j].append(sample)
                all_time_step_targets[j].append(target)
        # Re-batch the data by time steps across all validation batches
        self.eval_time_step_batches = [torch.cat(samples, dim=0) for samples in all_time_step_samples]
        self.eval_time_step_targets = [torch.cat(targets, dim=0) for targets in all_time_step_targets]


    def forward(self, x0=None, y=None, texts=None, eos=None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # Convert batch of sequences to list of indices
        # max_len = len(max(detokenized_samples, key=len))  # Find maximum sequence length in the batch
        # num_classes = len(self.mapping) - 1  # Number of classes is all except 'N'
        # # Create a tensor to hold the indices
        # batch_indices = torch.full((len(batch), max_len), self.mapping['N'], dtype=torch.long)  # Default to 'N'
        # for i, sequence in enumerate(batch):
        #     indices = torch.tensor([self.mapping.get(char, self.mapping['N']) for char in sequence], dtype=torch.long)
        #     batch_indices[i, :len(indices)] = indices
        # # One-hot encoding
        # onehot_tensor = F.one_hot(batch_indices, num_classes=num_classes).float()  # Exclude 'N' from the encoding
        # # Handle 'N' by zeroing out any positions that are supposed to be 'N'
        # n_mask = (batch_indices == self.mapping['N']).unsqueeze(-1)
        # onehot_tensor.masked_fill_(n_mask, 0.0)

        # x = self.embedding(x0)
        # x = self.head(x)
        # if x.shape != y.shape:
        #     x = x.squeeze(2)

        if self.training and not self.cdq:
            samples, mid_samples = self.ref_model._sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(samples)
            if self.task == "rna_saluki":
                target = self.reward_model(self.transform_samples_saluki(samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            onehot_mid_samples = [self.transform_samples(sample) for sample in mid_samples]
            onehot_mid_samples.append(onehot_samples)
            targets = [target for _ in range(len(onehot_mid_samples))]
            if self.timed:
                total_loss = 0
                for i, (sample, y) in enumerate(zip(onehot_mid_samples, targets)):
                    x = self.embedding(sample.float(), torch.full((sample.shape[0], sample.shape[1]), i).cuda())
                    x = self.head(x)
                    if x.shape != y.shape:
                        x = x.squeeze(2)
                    loss = self.loss_fct(x.view(-1), y.view(-1))
                    total_loss += loss

                return total_loss / len(onehot_mid_samples)

            x0 = torch.cat(onehot_mid_samples, dim=0)
            x0 = x0.float()
            y = torch.cat(targets, dim=0)
            # x0 = onehot_samples.float()
            # y = target
            x = self.embedding(x0)
            x = self.head(x)
            if x.shape != y.shape:
                x = x.squeeze(2)
            loss = self.loss_fct(x.view(-1), y.view(-1))
        elif self.training and self.cdq:
            samples, mid_samples, all_time_mid = self.ref_model._sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH, cdq=True)
            onehot_samples = self.transform_samples(samples)
            target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]

            appr_v = []
            for time, time_samples in enumerate(all_time_mid):
                if time == 0:
                    continue
                case_sum = 0
                for case in time_samples:
                    case_sum = case_sum + self.head(self.embedding(self.transform_samples(case).float())).squeeze(2).detach().clone()
                case_avg = case_sum / len(time_samples)

                # batch = torch.cat([self.transform_samples(case).float() for case in time_samples], dim=0)
                # # Pass the whole batch through the model
                # embeddings = self.embedding(batch)
                # outputs = self.head(embeddings).squeeze(2)
                # # Compute the mean of the outputs across the batch (dimension 0)
                # case_avg = torch.mean(outputs, dim=0)

                appr_v.append(case_avg.detach().clone())

            appr_v.append(target)
            onehot_mid_samples = [self.transform_samples(sample) for sample in mid_samples]
            onehot_mid_samples.append(onehot_samples)
            x0 = torch.cat(onehot_mid_samples, dim=0)
            x0 = x0.float()
            y = torch.cat(appr_v, dim=0)
            x = self.embedding(x0)
            x = self.head(x)
            if x.shape != y.shape:
                x = x.squeeze(2)
            loss = self.loss_fct(x.view(-1), y.view(-1))
        else:
            x = self.embedding(x0)
            x = self.head(x)
            if x.shape != y.shape:
                x = x.squeeze(2)
            loss = self.loss_fct(x.view(-1), y.view(-1))

        return loss

    def transform_samples(self, samples, num_classes=4):
        # One-hot encode the tensor but first mask out the '4's
        mask = samples != 4
        valid_samples = samples * mask
        one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)

        # Apply mask to zero out invalid rows
        one_hot_samples = one_hot_samples * mask.unsqueeze(-1)
        return one_hot_samples

    def transform_samples_saluki(self, samples, num_classes=4, final_length=12288):
        # One-hot encode the tensor but first mask out the '4's
        mask = samples != 4
        valid_samples = samples * mask
        one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)

        # Apply mask to zero out invalid rows
        one_hot_samples = one_hot_samples * mask.unsqueeze(-1)

        # Add two zero columns to each sample
        batch_size, seq_len, _ = one_hot_samples.shape
        padding_zeros = torch.zeros(batch_size, seq_len, 2, device=one_hot_samples.device, dtype=one_hot_samples.dtype)
        one_hot_samples = torch.cat((one_hot_samples, padding_zeros), dim=-1)
        if self.saluki_body == 6042:
            saluki_body = np.load(
                '/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/saluki_body_6042.npy')
        elif self.saluki_body == 2549:
            saluki_body = np.load(
                '/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/saluki_body_2549.npy')
        else:
            saluki_body = np.load('/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/saluki_body.npy')   #_2549, 6042
        saluki_body_tensor = torch.tensor(saluki_body, device=one_hot_samples.device, dtype=one_hot_samples.dtype)
        body_len = saluki_body_tensor.shape[0]
        # Concatenate saluki_body behind each sample
        one_hot_samples_with_body = torch.cat(
            (one_hot_samples, saluki_body_tensor.unsqueeze(0).repeat(batch_size, 1, 1)), dim=1
        )

        # Add zero padding to make the final output shape (batch_size, 12288, 6)
        padding_needed = final_length - one_hot_samples_with_body.shape[1]
        if padding_needed > 0:
            padding = torch.zeros(batch_size, padding_needed, 6, device=one_hot_samples.device,
                                  dtype=one_hot_samples.dtype)
            final_output = torch.cat((one_hot_samples_with_body, padding), dim=1)
        else:
            final_output = one_hot_samples_with_body[:, :final_length, :]

        return final_output

    @torch.no_grad()
    def evaluate_seq_step(self):
        # self.pearsonr.reset()
        losses = []
        pearsonr_scores = []
        for i, (batch, target) in enumerate(zip(self.eval_time_step_batches, self.eval_time_step_targets)):
            x0 = batch.detach().clone()
            y = target.detach().clone()
            x0 = x0.float()
            if self.timed:
                x = self.embedding(x0, torch.full((x0.shape[0], x0.shape[1]), i).cuda())
            else:
                x = self.embedding(x0)
            x = self.head(x)
            if x.shape != y.shape:
                x = x.squeeze(2)
            loss = self.loss_fct(x.view(-1), y.view(-1))
            losses.append(loss.item())
            # pearsonr_scores.append(self.pearsonr(y.view(-1), x.view(-1)).item())
            # Calculate Pearson correlation coefficient
            pearson_r, _ = pearsonr(x.detach().view(-1).cpu().numpy(), y.view(-1).cpu().numpy())
            pearsonr_scores.append(pearson_r)

        return losses, pearsonr_scores

    @torch.no_grad()
    def evaluation(self, batch_num):
        if self.task == "rna_old":
            self.reward_model = LightningModel.load_from_checkpoint(
                "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
                map_location='cpu')
        elif self.task == "rna":
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/RNA_evaluation:v0/model.ckpt",
                map_location='cpu')
        elif self.task == "rna_saluki":
            common_trunk = ConvGRUTrunk(
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
            self.reward_model = OriBaseModel(embedding=common_trunk, head=human_head)
            ckpt_human = torch.load('/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth', map_location='cpu')
            self.reward_model.load_state_dict(ckpt_human, strict=True)
        else:
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')

        self.reward_model.cuda()
        self.reward_model.eval()
        self.pearsonr.reset()
        targets = []
        predictions = []
        pearsonr_scores = []
        for i in range(batch_num):
            samples = self.ref_model.decode_sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(samples)
            if self.task == "rna_saluki":
                target = self.reward_model(self.transform_samples_saluki(samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            pred = self.head(self.embedding(onehot_samples.float())).squeeze(2).detach()
            targets.append(target)
            predictions.append(pred)
            pearsonr_scores.append(self.pearsonr(target.view(-1), pred.view(-1)).item())
        return predictions, targets, np.mean(pearsonr_scores)

    @torch.no_grad()
    def controlled_decode(self, gen_batch_num, sample_M):
        if self.task == "rna_old":
            self.reward_model = LightningModel.load_from_checkpoint(
                "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
                map_location='cpu')
        elif self.task == "rna":
            self.reward_model = LightningModel.load_from_checkpoint(
                  "artifacts/RNA_evaluation:v0/model.ckpt",
                map_location='cpu')
        elif self.task == "rna_saluki":
            common_trunk = ConvGRUTrunk(
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
            self.reward_model = OriBaseModel(embedding=common_trunk, head=human_head)
            ckpt_human = torch.load('/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth', map_location='cpu')
            self.reward_model.load_state_dict(ckpt_human, strict=True)
        else:
            self.reward_model = LightningModel.load_from_checkpoint(
                 "artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')

        self.reward_model.cuda()
        self.reward_model.eval()
        samples = []
        value_func_preds = []
        reward_model_preds = []
        for i in range(gen_batch_num):
            batch_samples = self.ref_model.controlled_sample(self.embedding, self.head, eval_sp_size=self.NUM_SAMPLES_PER_BATCH, sample_M=sample_M)
            samples.append(batch_samples)
            onehot_samples = self.transform_samples(batch_samples)
            value_func_preds.extend(self.head(self.embedding(onehot_samples.float())).squeeze(2).detach())
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch_samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            reward_model_preds.extend(pred)

        print("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num*sample_M):
            batch = self.ref_model.decode_sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(batch)
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            if i < gen_batch_num:
                baseline_preds.extend(pred)
            all_preds.extend(pred)

        print("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        top_k_values, _ = torch.topk(all_values, k)

        return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)


    def controlled_decode_DPS(self, gen_batch_num, sample_M):
        if self.task == "rna_old":
            self.reward_model = LightningModel.load_from_checkpoint(
                "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
                map_location='cpu')
        elif self.task == "rna":
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/RNA_evaluation:v0/model.ckpt",
                map_location='cpu')
        elif self.task == "rna_saluki":
            common_trunk = ConvGRUTrunk(
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
            self.reward_model = OriBaseModel(embedding=common_trunk, head=human_head)
            ckpt_human = torch.load('/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth', map_location='cpu')
            self.reward_model.load_state_dict(ckpt_human, strict=True)
        else:
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')

        self.reward_model.cuda()
        self.reward_model.eval()
        samples = []
        value_func_preds = []
        reward_model_preds = []
        for i in range(gen_batch_num):
            batch_samples = self.ref_model.controlled_sample_DPS(self.embedding, self.head, eval_sp_size=self.NUM_SAMPLES_PER_BATCH, sample_M=sample_M)
            samples.append(batch_samples)
            onehot_samples = self.transform_samples(batch_samples)
            value_func_preds.extend(self.head(self.embedding(onehot_samples.float())).squeeze(2).detach())
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch_samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            reward_model_preds.extend(pred)

        print("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num*sample_M):
            batch = self.ref_model.decode_sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(batch)
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            if i < gen_batch_num:
                baseline_preds.extend(pred)
            all_preds.extend(pred)

        print("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        top_k_values, _ = torch.topk(all_values, k)

        return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)
    
    @torch.no_grad()
    def controlled_decode_tweedie(self, gen_batch_num, sample_M, options):
        if self.task == "rna_old":
            self.reward_model = LightningModel.load_from_checkpoint(
                "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
                map_location='cpu')
        elif self.task == "rna":
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/RNA_evaluation:v0/model.ckpt",
                map_location='cpu')
        elif self.task == "rna_saluki":
            common_trunk = ConvGRUTrunk(
                stem_channels=64,
                stem_kernel_size=15,
                n_conv=6,
                channel_init=64,
                channel_mult=1,
                kernel_size=5,
                act_func="relu",
                conv_norm=True,
                pool_func=None,  # None, "max", "avg"
                pool_size=None,
                residual=True,  # False
                crop_len=0,
                n_gru=1,
                dropout=0.1,  # 0.3
                gru_norm=True, )
            human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
            self.reward_model = OriBaseModel(embedding=common_trunk, head=human_head)
            ckpt_human = torch.load('/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth', map_location='cpu')
            self.reward_model.load_state_dict(ckpt_human, strict=True)
        else:
            self.reward_model = LightningModel.load_from_checkpoint(
                "artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')

        self.reward_model.cuda()
        self.reward_model.eval()


        samples = []
        value_func_preds = []
        reward_model_preds = []
        for i in range(gen_batch_num):
            batch_samples = self.ref_model.controlled_sample_tweedie(self.reward_model, eval_sp_size=self.NUM_SAMPLES_PER_BATCH, sample_M=sample_M, options = options, task=self.task)
            samples.extend(batch_samples)
            onehot_samples = self.transform_samples(batch_samples)
            value_func_preds.extend(self.head(self.embedding(onehot_samples.float())).squeeze(2).detach())
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch_samples).float()).detach().squeeze(2)
            elif self.n_tasks==1:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
            reward_model_preds.extend(pred)

        print("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num*sample_M):
            batch = self.ref_model.decode_sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(batch)
            if self.task == "rna_saluki":
                pred = self.reward_model(self.transform_samples_saluki(batch).float()).detach().squeeze(2)
            elif self.n_tasks == 1 :
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            else:
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()
                print("1")
                '''
                threshold = 0.8
                reward_1 = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 1]
                reward_2 = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 2]
                reward_pes1 = torch.clamp(5.0*(threshold - reward_1),max=1.0)
                reward_pes2= torch.clamp(5.0*(threshold - reward_2),max=1.0)
                torch.clamp(self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 1])
              
                pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0] +  torch.log(torch.clamp(reward_pes1,min= 1e-40) ) + torch.log(torch.clamp(reward_pes2,min= 1e-40) ) 
                '''
            if i < gen_batch_num:
                baseline_preds.extend(pred)
            all_preds.extend(pred)

        top_k_values = torch.cat(baseline_preds)
        '''
        print("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        top_k_values, _ = torch.topk(all_values, k)
        '''

        return samples, torch.cat(value_func_preds), torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if 'ref_model' not in fpn and 'reward_model' not in fpn:
                    no_decay.add(fpn)

                # if pn.endswith('bias') or ('bias' in pn):
                #     # all biases will not be decayed
                #     no_decay.add(fpn)
                # else:
                #     if (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                #         # weights of whitelist modules will be weight decayed
                #         decay.add(fpn)
                #     else:
                #         if (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, blacklist_weight_modules):
                #             # weights of blacklist modules will NOT be weight decayed
                #             no_decay.add(fpn)
                #         else:
                #             no_decay.add(fpn)
                # # elif pn.endswith('weight') and (('norm' in pn) or ('embedding' in pn)):
                # #     no_decay.add(fpn)
                # # else:
                # #     no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )
        if len(param_dict.keys() - union_params) != 0:
            print(f"skipping param: {param_dict.keys() - union_params}")

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 'lr': train_config.learning_rate * 2,  "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


class BaseModelMultiSep(nn.Module):
    """
    Base model class
    """

    def __init__(self, embedding: nn.Module, head: nn.Module, cdq, batch_size, val_batch_num) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head
        self.embedding1 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head1 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding2 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head2 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding3 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head3 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding4 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head4 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding5 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head5 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding6 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head6 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding7 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head7 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding8 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head8 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        self.embedding9 = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                        attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        self.head9 = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')

        self.loss_fct = nn.MSELoss()
        self.pearsonr = PearsonR(num_targets=1)
        self.cdq = cdq
        # self.mapping = {"A": 0, "C": 1, "G": 2, "T": 3} N: 4
        # self.num_features = len(self.mapping)

        CKPT_PATH = '/home/lix361/projects/rna_optimization/seqft2/gosai_dna/last.ckpt'
        self.NUM_SAMPLES_PER_BATCH = batch_size
        # reinitialize Hydra
        GlobalHydra.instance().clear()
        # Initialize Hydra and compose the configuration
        initialize(config_path="configs_gosai", job_name="load_model")
        cfg = compose(config_name="config_gosai.yaml")
        # Initialize the model
        self.ref_model = diffusion_gosai.Diffusion.load_from_checkpoint(CKPT_PATH, config=cfg, map_location='cpu')
        # self.detokenizer = dataloader_gosai.DNASequenceDetokenizer()

        # self.ref_model.load_state_dict(torch.load(ref_model_path, map_location='cpu')['model_state_dict'], strict=True)
        # self.tokenizer = tokenizer
        self.ref_model.cuda()
        self.ref_model.eval()
        # Freeze the ref_model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.reward_model = LightningModel.load_from_checkpoint("artifacts/DNA_evaluation:v0/model.ckpt", map_location='cpu')
        self.reward_model.cuda()
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.val_data_num = val_batch_num * batch_size
        # Initialize lists to store all one-hot samples and targets by time steps across all validation batches
        all_time_step_samples = [[] for _ in range(128)]
        all_time_step_targets = [[] for _ in range(128)]
        for i in range(val_batch_num):
            samples, mid_samples = self.ref_model._sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(samples)
            target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            onehot_mid_samples = [self.transform_samples(sample) for sample in mid_samples]
            onehot_mid_samples.append(onehot_samples)
            # targets = [target for _ in range(len(onehot_mid_samples))]
            # x0 = torch.cat(onehot_mid_samples, dim=0)
            # x0 = x0.float()
            # y = torch.cat(targets, dim=0)
            # Store samples and targets in corresponding time step lists
            for j, sample in enumerate(onehot_mid_samples):
                all_time_step_samples[j].append(sample)
                all_time_step_targets[j].append(target)

        # Re-batch the data by time steps across all validation batches
        self.eval_time_step_batches = [torch.cat(samples, dim=0) for samples in all_time_step_samples]
        self.eval_time_step_targets = [torch.cat(targets, dim=0) for targets in all_time_step_targets]


    def forward(self, x0=None, y=None, texts=None, eos=None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """

        if self.training and not self.cdq:
            samples, mid_samples = self.ref_model._sample(eval_sp_size=self.NUM_SAMPLES_PER_BATCH)
            onehot_samples = self.transform_samples(samples)
            target = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            onehot_mid_samples = [self.transform_samples(sample) for sample in mid_samples]
            onehot_mid_samples.append(onehot_samples)
            targets = [target for _ in range(len(onehot_mid_samples))]
            losses = []
            current_multimodel_loss_sum = []
            current = 0
            total_loss = 0
            for i, (sample, y) in enumerate(zip(onehot_mid_samples, targets)):
                embedding, head, index = self.get_model_components(i)
                if index != current:
                    losses.append(np.mean(current_multimodel_loss_sum))
                    current_multimodel_loss_sum = []
                    current = copy.copy(index)
                x = embedding(sample.float())
                x = head(x)
                if x.shape != y.shape:
                    x = x.squeeze(2)
                loss = self.loss_fct(x.view(-1), y.view(-1))
                total_loss += loss
                current_multimodel_loss_sum.append(loss.item())

            losses.append(np.mean(current_multimodel_loss_sum))
            assert len(losses) == 10

        return total_loss/len(onehot_mid_samples), losses

    def transform_samples(self, samples, num_classes=4):
        # One-hot encode the tensor but first mask out the '4's
        mask = samples != 4
        valid_samples = samples * mask
        one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)

        # Apply mask to zero out invalid rows
        one_hot_samples = one_hot_samples * mask.unsqueeze(-1)
        return one_hot_samples

    def evaluate_seq_step(self):
        self.pearsonr.reset()
        losses = []
        pearsonr_scores = []
        for i, (batch, target) in enumerate(zip(self.eval_time_step_batches, self.eval_time_step_targets)):
            embedding, head, _ = self.get_model_components(i)
            x0 = batch.detach().clone()
            y = target.detach().clone()
            x0 = x0.float()
            x = embedding(x0)
            x = head(x)
            if x.shape != y.shape:
                x = x.squeeze(2)
            loss = self.loss_fct(x.view(-1), y.view(-1))
            losses.append(loss.item())
            pearsonr_scores.append(self.pearsonr(y.view(-1), x.view(-1)).item())
        return losses, pearsonr_scores

    def get_model_components(self, time_step):
        if time_step < 20:
            return self.embedding, self.head, 0
        index = (time_step - 20) // 12
        if index == 0:
            return self.embedding1, self.head1, index+1
        elif index == 1:
            return self.embedding2, self.head2, index+1
        elif index == 2:
            return self.embedding3, self.head3, index+1
        elif index == 3:
            return self.embedding4, self.head4, index+1
        elif index == 4:
            return self.embedding5, self.head5, index+1
        elif index == 5:
            return self.embedding6, self.head6, index+1
        elif index == 6:
            return self.embedding7, self.head7, index+1
        elif index == 7:
            return self.embedding8, self.head8, index+1
        elif index == 8:
            return self.embedding9, self.head9, index+1
        else:
            raise ValueError("Time step out of expected range")

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if 'ref_model' not in fpn and 'reward_model' not in fpn:
                    no_decay.add(fpn)

                # if pn.endswith('bias') or ('bias' in pn):
                #     # all biases will not be decayed
                #     no_decay.add(fpn)
                # else:
                #     if (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                #         # weights of whitelist modules will be weight decayed
                #         decay.add(fpn)
                #     else:
                #         if (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, blacklist_weight_modules):
                #             # weights of blacklist modules will NOT be weight decayed
                #             no_decay.add(fpn)
                #         else:
                #             no_decay.add(fpn)
                # # elif pn.endswith('weight') and (('norm' in pn) or ('embedding' in pn)):
                # #     no_decay.add(fpn)
                # # else:
                # #     no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )
        if len(param_dict.keys() - union_params) != 0:
            print(f"skipping param: {param_dict.keys() - union_params}")

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 'lr': train_config.learning_rate * 2,  "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


class OriBaseModel(nn.Module):
    """
    Base model class
    """

    def __init__(self, embedding: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.embedding(x)
        x = self.head(x)
        return x


class EnformerModel(BaseModel):
    """
    Enformer model architecture.

    Args:
        n_tasks: Number of tasks for the model to predict
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
        final_act_func: Name of the activation function to use in the final layer
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
        # Head
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = "avg",
    ) -> None:
        # super().__init__(
        embedding=EnformerTrunk(
            n_conv=n_conv,
            channels=channels,
            n_transformers=n_transformers,
            n_heads=n_heads,
            key_len=key_len,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
            crop_len=crop_len,
        )
        head=ConvHead(
            n_tasks=n_tasks,
            in_channels=2 * channels,
            act_func=final_act_func,
            norm=False,
            pool_func=final_pool_func,
        )
        # )
        super().__init__(embedding, head)


class TimeEmbedding(nn.Module):
    def __init__(self, max_time_steps, embedding_size):
        super().__init__()
        self.time_embedding = nn.Embedding(max_time_steps, embedding_size)

    def forward(self, time_indices):
        return self.time_embedding(time_indices)


class TimedEnformerTrunk(nn.Module):
    """
    Enformer model architecture.

    Args:
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
    """

    def __init__(
        self,
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
    ) -> None:
        super().__init__()

        self.conv_tower = EnformerConvTower(n_blocks=n_conv, out_channels=channels)
        self.transformer_tower = EnformerTransformerTower(
            in_channels=channels,
            n_blocks=n_transformers,
            n_heads=n_heads,
            key_len=key_len,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
        )
        self.pointwise_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
            act_func="gelu_enformer",
            dropout=ff_dropout // 8,
            order="NACDR",
        )
        self.act = Activation("gelu_enformer")
        self.crop = Crop(crop_len)
        self.time_embedding = TimeEmbedding(max_time_steps=128, embedding_size=4)

    def forward(self, x, time_indices):
        time_embeds = self.time_embedding(time_indices)  # Shape: [batch_size, seq_length, channels]
        x = x + 0.01*time_embeds
        # Adjust the input dimension
        x = x.transpose(1, 2)  # Transpose the dimensions to [batch_size, features, seq_length]
        x = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x


class EnformerTrunk(nn.Module):
    """
    Enformer model architecture.

    Args:
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
    """

    def __init__(
        self,
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
    ) -> None:
        super().__init__()

        self.conv_tower = EnformerConvTower(n_blocks=n_conv, out_channels=channels)
        self.transformer_tower = EnformerTransformerTower(
            in_channels=channels,
            n_blocks=n_transformers,
            n_heads=n_heads,
            key_len=key_len,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
        )
        self.pointwise_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
            act_func="gelu_enformer",
            dropout=ff_dropout // 8,
            order="NACDR",
        )
        self.act = Activation("gelu_enformer")
        self.crop = Crop(crop_len)

    def forward(self, x):
        # Adjust the input dimension
        x = x.transpose(1, 2)  # Transpose the dimensions to [batch_size, features, seq_length]
        x = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x


class ConvGRUTrunk(nn.Module):
    """
    A model consisting of a convolutional tower followed by a bidirectional GRU layer and optional pooling.

    Args:
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        conv_norm: If True, apply batch normalization in the convolutional layers.
        residual: If True, apply residual connections in the convolutional layers.
        crop_len: Number of positions to crop at either end of the output
        n_gru: Number of GRU layers
        dropout: Dropout for GRU and feed-forward layers
        gru_norm: If True, include layer normalization in feed-forward network.
    """

    def __init__(
        self,
        # Stem
        stem_in_channels: int = 6,
        stem_channels: int = 16,
        stem_kernel_size: int = 15,
        # Conv
        n_conv: int = 2,
        channel_init: int = 16,
        channel_mult: float = 1,
        kernel_size: int = 5,
        act_func: str = "relu",
        conv_norm: bool = False,
        pool_func: Optional[str] = None,
        pool_size: Optional[int] = None,
        residual: bool = False,
        # Crop
        crop_len: int = 0,
        # GRU
        n_gru: int = 1,
        dropout: float = 0.0,
        gru_norm: bool = False,
    ):
        super().__init__()
        self.conv_tower = ConvTower(
            stem_in_channels=stem_in_channels,
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_blocks=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            dilation_init=1,
            dilation_mult=1,
            act_func=act_func,
            norm=conv_norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            dropout= dropout,
            order="CDNRA",
            crop_len=crop_len,
        )

        self.gru_tower = GRUBlock(
            in_channels=self.conv_tower.out_channels,
            n_layers=n_gru,
            dropout=dropout,
            act_func=act_func,
            norm=gru_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # Adjust the input dimension
        if x.shape[1] != self.conv_tower.blocks[0].conv.in_channels:
            x = x.transpose(1, 2)  # Transpose the dimensions to [batch_size, features, seq_length]
        x = self.conv_tower(x)
        x = self.gru_tower(x)
        return x


class dilated_residual(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, dilation=1, dropout=0):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            channels,
            kernel_size=kernel_size,
            act_func="gelu",
            activation="first",
            norm=True,
            residual=False,
            dilation=dilation,
        )
        self.conv2 = ConvBlock(
            channels,
            in_channels,
            kernel_size=kernel_size,
            act_func="gelu",
            activation="first",
            norm=True,
            residual=False,
            dilation=1,
            dropout=dropout,
        )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = x + inputs
        return x


class dilated_residual_tower(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        n_blocks=1,
        kernel_size=3,
        dilation_mult=2,
        dropout=0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.receptive_field = 0
        dilation = 1
        for i in range(n_blocks):
            self.layers.append(
                dilated_residual(
                    in_channels,
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            self.receptive_field += (dilation + 1) * (kernel_size - 1)
            dilation = int(dilation * dilation_mult)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Basenji(nn.Module):
    def __init__(
        self,
        n_tasks,
        conv_blocks=4,
        channel_init=256,
        kernel_size=5,
        pool_func="max",
        conv_dropout=0.05,
        residual_channels=108,
        residual_blocks=6,
        residual_dropout=0.1,
        conv_channel_mult=1.125,
        dilation_mult=1.2,
        crop_len=0,
        final_pool_func="avg",
    ):
        super().__init__()

        self.n_tasks = n_tasks

        self.conv_tower = ConvTower(
            n_blocks=conv_blocks,
            channel_init=channel_init,
            channel_mult=conv_channel_mult,
            stem_channels=channel_init,
            stem_kernel_size=15,
            kernel_size=kernel_size,
            dilation_init=1,
            dilation_mult=1,
            act_func="gelu",
            norm=True,
            pool_func=pool_func,
            residual_skip=1,
            activation="last",
            dropout=conv_dropout,
            crop_len=0,
        )

        self.dilated_residual_tower = dilated_residual_tower(
            in_channels=self.conv_tower.out_channels,
            channels=residual_channels,
            kernel_size=kernel_size,
            dilation_mult=dilation_mult,
            n_blocks=residual_blocks,
            dropout=residual_dropout,
        )

        self.conv2 = ConvBlock(
            self.conv_tower.out_channels,
            self.conv_tower.out_channels,
            kernel_size=1,
            dropout=conv_dropout,
        )

        self.crop = Crop(
            crop_len,
            receptive_field=self.conv_tower.receptive_field
            + self.dilated_residual_tower.receptive_field,
        )
        self.head = ChannelTransform(self.conv_tower.out_channels, self.n_tasks)
        self.pool = AdaptivePool(final_pool_func)

    def embed(self, x):
        x = self.conv_tower(x)  # N, 64, L//128
        x = self.dilated_residual_tower(x)  # N, 64, L//128
        x = self.crop(x)  # N, n_tasks, out_len
        x = self.conv2(x)  # N, 64, L//128
        return x

    def forward(self, x):  # N, 4, L
        x = self.embed(x)  # N, 64, L//128
        x = self.head(x)  # N, n_tasks, L//128
        x = self.pool(x)  # N, n_tasks, 1
        return x


class GRUBlock(nn.Module):
    """
    Stacked bidirectional GRU layers followed by a feed-forward network.

    Args:
        in_channels: The number of channels in the input
        n_layers: The number of GRU layers
        gru_hidden_size: Number of hidden elements in GRU layers
        dropout: Dropout probability
        act_func: Name of the activation function for feed-forward network
        norm: If True, include layer normalization in feed-forward network.

    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        act_func: str = "relu",
        norm: bool = False,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=in_channels,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
            num_layers=n_layers,
        )
        self.ffn = FeedForwardBlock(
            in_len=in_channels, dropout=dropout, act_func=act_func
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        x = self.gru(x)[0]
        # Combine output of forward and reverse GRU
        x = x[:, :, : self.gru.hidden_size] + x[:, :, self.gru.hidden_size :]  # saluki comment out

        # Output shape will be [batch_size, seq_length, 2 * in_channels]
        # Extracting only the outputs from the backward pass
        # output, hidden = self.gru(x)
        # x = x[:, :, self.gru.hidden_size:]    # saluki use
        # x = x[:, 0, self.gru.hidden_size:].unsqueeze(1)  # saluki use

        x = self.ffn(x)
        x = rearrange(x, "b l t -> b t l")
        return x



class ConvTower(nn.Module):
    """
    A module that consists of multiple convolutional blocks and takes a one-hot encoded
    DNA sequence as input.

    Args:
        n_blocks: Number of convolutional blocks, including the stem
        stem_channels: Number of channels in the stem,
        stem_kernel_size: Kernel width for the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        dilation_init: Initial dilation
        dilation_mult: Factor by which to multiply the dilation in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        dropout: Dropout probability
        norm: If True, apply batch norm
        residual: If True, apply residual connection
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNRA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, residual addition, activation. Pooling is not included
            as it is always performed last.
        crop_len: Number of positions to crop at either end of the output
    """

    def __init__(
        self,
        stem_in_channels: int,
        stem_channels: int,
        stem_kernel_size: int,
        n_blocks: int = 2,
        channel_init: int = 16,
        channel_mult: float = 1,
        kernel_size: int = 5,
        dilation_init: int = 1,
        dilation_mult: float = 1,
        act_func: str = "relu",
        norm: bool = False,
        pool_func: Optional[str] = None,
        pool_size: Optional[int] = None,
        residual: bool = False,
        dropout: float = 0.0,
        order: str = "CDNRA",
        crop_len: Union[int, str] = 0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(Stem(stem_in_channels, stem_channels, stem_kernel_size, act_func=act_func))
        self.receptive_field = stem_kernel_size
        self.pool_factor = 1
        self.out_channels = stem_channels

        # Add the remaining n-1 blocks
        in_channels = stem_channels
        out_channels = channel_init
        dilation = dilation_init

        for i in range(1, n_blocks):
            # Add block
            self.blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    act_func=act_func,
                    norm=norm,
                    residual=residual,
                    pool_func=pool_func,
                    pool_size=pool_size,
                    dropout=dropout,
                    order=order,
                )
            )

            # Account for kernel width
            self.receptive_field += dilation * (kernel_size - 1)

            # Account for pooling
            if pool_func is not None:
                self.receptive_field *= pool_size
                self.pool_factor *= pool_size

            # Set final number of output channels
            if i == n_blocks - 1:
                self.out_channels = out_channels

            else:
                # Output channels of this block become the input channels of the next block
                in_channels = out_channels

                # Multiply output channels and dilation
                out_channels = int(out_channels * channel_mult)
                dilation = int(dilation * dilation_mult)

        # Cropping layer
        self.crop = Crop(crop_len, receptive_field=self.receptive_field)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks:
            x = block(x)
        x = self.crop(x)
        return x


class Stem(nn.Module):
    """
    Convolutional layer followed by optional activation and pooling.
    Meant to take one-hot encoded DNA sequence as input

    Args:
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of pooling layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,   #track
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            bias=True,
        )
        self.act = Activation(act_func)
        self.pool = Pool(pool_func, pool_size=pool_size)
        self.norm = Norm("layer", in_dim=out_channels)  # saluki, layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        # x = self.norm(x)  # saluki
        x = self.act(x)
        x = self.pool(x)
        return x


class EnformerConvTower(nn.Module):
    """
    Args:
        n_blocks: Number of convolutional/pooling blocks including the stem.
        out_channels: Number of channels in the output
    """

    def __init__(
        self,
        n_blocks: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        half_dim = out_channels // 2

        # Empty list
        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(4, half_dim, 15, padding="same"),   #track
                ConvBlock(
                    in_channels=half_dim,
                    out_channels=half_dim,
                    kernel_size=1,
                    act_func="gelu_enformer",
                    residual=True,
                    order="NACDR",
                    pool_func="attn",
                    pool_size=2,
                ),
            )
        )

        # List input and output channels for the remaining n_blocks - 1 blocks
        filters = [half_dim] + exponential_linspace_int(
            half_dim, out_channels, num=(n_blocks - 1), divisible_by=128
        )

        # Add the remaining n_blocks - 1 blocks
        for i in range(1, n_blocks):
            self.blocks.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels=filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=5,
                        act_func="gelu_enformer",
                        residual=False,
                        order="NACDR",
                    ),
                    ConvBlock(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=1,
                        act_func="gelu_enformer",
                        residual=True,
                        order="NACDR",
                        pool_func="attn",
                        pool_size=2,
                    ),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks:
            x = block(x)
        return x


class EnformerTransformerBlock(nn.Module):
    """
    Transformer tower for enformer model

    Args:
        in_len: Length of the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
    """

    def __init__(
        self,
        in_len: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        self.norm = Norm("layer", in_len)
        self.mha = Attention(
            dim=in_len,
            heads=n_heads,
            dim_key=key_len,
            dim_value=in_len // n_heads,
            dropout=attn_dropout,
            pos_dropout=pos_dropout,
            num_rel_pos_features=in_len // n_heads,
            use_tf_gamma=False,
        )
        self.dropout = Dropout(ff_dropout)
        self.ffn = FeedForwardBlock(
            in_len=in_len,
            dropout=ff_dropout,
            act_func="relu",
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = torch.add(ffn_input, x)
        return x


class EnformerTransformerTower(nn.Module):
    """
    Transformer tower for enformer model

    Args:
        in_channels: Number of channels in the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
    """

    def __init__(
        self,
        in_channels: int,
        n_blocks: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EnformerTransformerBlock(
                    in_len=in_channels,
                    n_heads=n_heads,
                    key_len=key_len,
                    attn_dropout=attn_dropout,
                    pos_dropout=pos_dropout,
                    ff_dropout=ff_dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, "b l t -> b t l")
        return x


class FeedForwardBlock(nn.Module):
    """
    2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

    Args:
        in_len: Length of the input tensor
        dropout: Dropout probability
        act_func: Name of the activation function
    """

    def __init__(
        self, in_len: int, dropout: float = 0.0, act_func: str = "relu"
    ) -> None:
        super().__init__()
        self.dense1 = LinearBlock(
            in_len, in_len * 2, norm=True, dropout=dropout, act_func=act_func, bias=True
        )
        self.dense2 = LinearBlock(
            in_len * 2, in_len, norm=False, dropout=dropout, act_func=None, bias=True
        )
        self.dense = LinearBlock(
            in_len, in_len, norm=True, dropout=dropout, act_func=act_func, bias=True
        )        # saluki

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.dense1(x)
        x = self.dense2(x)
        # x = self.dense(x)  # saluki
        return x


class LinearBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout.

    Args:
        in_len: Length of input
        out_len: Length of output
        act_func: Name of activation function
        dropout: Dropout probability
        norm: If True, apply layer normalization
        bias: If True, include bias term.
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        act_func: str = "relu",
        dropout: float = 0.0,
        norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.norm = Norm(func="layer" if norm else None, in_dim=in_len)  # layer, saluki use batch
        self.linear = nn.Linear(in_len, out_len, bias=bias)
        self.dropout = Dropout(dropout)
        self.act = Activation(act_func)
        # self.norm1 = Norm(func="batch" if norm else None, in_dim=out_len)  # saluki
        # self.act1 = Activation(act_func)  # saluki

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        # x = self.act(x)  # saluki use
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)  # saluki comment out
        # x = self.norm1(x)  # saluki add
        # x = self.act1(x)  # saluki add
        return x


class Head(nn.Module):  #saluki
    """
    Linear layer
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_len, out_len, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.linear(x.squeeze())
        return x


class ConvHead(nn.Module):
    """
    A 1x1 Conv layer that transforms the number of channels in the input and then
    optionally pools along the length axis.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the convolutional layer
        pool_func: Pooling function.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        act_func: Optional[str] = None,
        pool_func: Optional[str] = None,
        norm: bool = False,
    ) -> None:
        super().__init__()
        # Save all params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.act_func = act_func
        self.pool_func = pool_func
        self.norm = norm

        # Create layers
        self.channel_transform = ChannelTransformBlock(
            self.in_channels, self.n_tasks, act_func=self.act_func, norm=self.norm
        )
        self.pool = AdaptivePool(self.pool_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        x = self.channel_transform(x)
        x = self.pool(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional layer along with optional normalization,
    activation, dilation, dropout, residual connection, and pooling.
    The order of these operations can be specified, except
    for pooling, which always comes last.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        dilation: Dilation
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Pooling width
        dropout: Dropout probability
        norm: If True, apply batch norm
        residual: If True, apply residual connection
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNRA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, residual addition, activation. Pooling is not included
            as it is always performed last.
        return_pre_pool: If this is True and pool_func is not None, the final
            output will be a tuple (output after pooling, output_before_pooling).
            This is useful if the output before pooling is required by a later
            layer.
        **kwargs: Additional arguments to be passed to nn.Conv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
        dropout: float = 0.0,
        norm: bool = True,
        residual: bool = False,
        order: str = "CDNRA",
        bias: bool = True,
        return_pre_pool: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
            "R",
        ], "The string supplied in order must contain one occurrence each of A, C, D, N and R."
        self.order = order

        # Create batch norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm("batch", in_dim=out_channels)  # batch, saluki use layer
            else:
                self.norm = Norm("batch", in_dim=in_channels)   # batch, saluki use layer
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            **kwargs,
        )
        self.act = Activation(act_func)
        self.pool = Pool(func=pool_func, pool_size=pool_size, in_channels=out_channels)
        self.dropout = Dropout(dropout)
        self.residual = residual
        if self.residual:
            self.channel_transform = ChannelTransform(in_channels, out_channels)
        self.order = order
        assert (
            len(set(self.order).difference(set("CDNRA"))) == 0
        ), "The string supplied in order contains a non-recognized letter."
        self.return_pre_pool = return_pre_pool

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input data.
        """
        if self.residual:
            x_input = self.channel_transform(x)

        # Intermediate layers
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "R":
                if self.residual:
                    x = torch.add(x, x_input)
            elif name == "A":
                x = self.act(x)

        # Pool
        if self.return_pre_pool:
            return self.pool(x), x
        else:
            return self.pool(x)


class ChannelTransformBlock(nn.Module):
    """
    Convolutional layer with kernel size=1 along with optional normalization, activation
    and dropout

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        act_func: Name of the activation function
        dropout: Dropout probability
        norm: If True, apply batch norm
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, activation.
        if_equal: If True, create a layer even if the input and output channels are equal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
        act_func: str = "relu",
        dropout: float = 0.0,
        order: str = "CDNA",
        if_equal: bool = False,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
        ], "The string supplied in order must contain one occurrence each of A, C, D and N."
        self.order = order

        # Create batch norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm("batch", in_dim=out_channels)
            else:
                self.norm = Norm("batch", in_dim=in_channels)
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = ChannelTransform(in_channels, out_channels, if_equal=if_equal)
        self.act = Activation(act_func)
        self.dropout = Dropout(dropout)
        self.order = order

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "A":
                x = self.act(x)
        return x


class Activation(nn.Module):
    """
    A nonlinear activation layer.

    Args:
        func: The type of activation function. Supported values are 'relu',
            'elu', 'softplus', 'gelu', 'gelu_enformer' and 'exp'. If None, will return nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported activation function.
    """

    def __init__(self, func: str) -> None:
        super().__init__()

        if func == "relu":
            self.layer = nn.ReLU()
        elif func == "elu":
            self.layer = nn.ELU()
        elif func == "gelu":
            self.layer = nn.GELU()
        elif func == "gelu_enformer":
            self.layer = GELU()
        elif func == "softplus":
            self.layer = nn.Softplus()
        elif func == "exp":
            self.layer = torch.exp
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Pool(nn.Module):
    """
    A pooling layer.

    Args:
        func: Type of pooling function. Supported values are 'avg', 'max',
            or 'attn'. If None, will return nn.Identity.
        pool_size: The number of positions to pool together
        in_channels: Number of channels in the input. Only needeed for attention pooling.
        **kwargs: Additional arguments to pass to the pooling function.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(
        self,
        func: Optional[str],
        pool_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if func == "avg":
            self.layer = nn.AvgPool1d(kernel_size=pool_size, **kwargs)
        elif func == "max":
            self.layer = nn.MaxPool1d(kernel_size=pool_size, **kwargs)
        elif func == "attn":
            if in_channels is None:
                raise ValueError("The number of input channels must be provided.")
            self.layer = AttentionPool(dim=in_channels, pool_size=pool_size, **kwargs)
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class AdaptivePool(nn.Module):
    """
    An Adaptive Pooling layer. This layer does not have a defined pooling width but
    instead pools together all the values in the last axis.

    Args:
        func: Type of pooling function. Supported values are 'avg' or 'max'. If None,
            will return nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(self, func: Optional[str] = None) -> None:
        super().__init__()

        if func == "avg":
            self.layer = nn.AdaptiveAvgPool1d(1)
        elif func == "max":
            self.layer = nn.AdaptiveMaxPool1d(1)
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Norm(nn.Module):
    """
    A batch normalization or layer normalization layer.

    Args:
        func: Type of normalization function. Supported values are 'batch' or 'layer'. If None,
            will return nn.Identity.
        in_dim: Number of features in the input tensor.
        **kwargs: Additional arguments to pass to the normalization function.
    """

    def __init__(
        self, func: Optional[str] = None, in_dim: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__()
        self.func = func
        self.in_dim = in_dim
        if func == "batch":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.BatchNorm1d(in_dim, **kwargs)

        elif func == "layer":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.LayerNorm(in_dim, **kwargs)

        elif func is None:
            self.layer = nn.Identity()

        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        if self.func == "layer" and x.shape[2] != self.in_dim:
            x = x.transpose(1, 2)  # Transpose the dimensions to [batch_size, seq_length, features]
            x_n = self.layer(x)
            x_n = x_n.transpose(1, 2)
            return x_n
        elif self.func == "batch" and x.shape[1] != self.in_dim:
            x = x.transpose(1, 2)
            x_n = self.layer(x)
            x_n = x_n.transpose(1, 2)
            return x_n
        else:
            return self.layer(x)


class ChannelTransform(nn.Module):
    """
    A convolutional layer to transform the number of channels in the input.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        if_equal: Whether to create layer if input and output channels are equal
        **kwargs: Additional arguments to pass to the convolutional layer.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 1, if_equal: bool = False, **kwargs
    ) -> None:
        super().__init__()
        if (in_channels == out_channels) and (not if_equal):
            self.layer = nn.Identity()
        else:
            self.layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same", **kwargs
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Dropout(nn.Module):
    """
    Optional dropout layer

    Args:
        p: Dropout probability. If this is set to 0, will return nn.Identity.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Crop(nn.Module):
    """
    Optional cropping layer.

    Args:
        crop_len: Number of positions to crop at each end of the input.
        receptive_field: Receptive field of the model to calculate crop_len.
            Only needed if crop_len is None.
    """

    def __init__(
        self, crop_len: int = 0, receptive_field: Optional[int] = None
    ) -> None:
        super().__init__()
        if crop_len == 0:
            self.layer = nn.Identity()
        else:
            if crop_len == "auto":
                assert (
                    receptive_field is not None
                ), "Receptive field must be provided for autocropping"
                # crop_len = int(np.ceil(receptive_field / 2))
                crop_len = int(receptive_field // 2)
            self.layer = nn.ConstantPad1d(-crop_len, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


# class Attention(nn.Module):
#     def __init__(
#         self,
#         in_len: int,
#         key_len: int,
#         value_len: int,
#         n_heads: int,
#         n_pos_features: int,
#         pos_dropout: float = 0,
#         attn_dropout: float = 0,
#     ):
#         """
#         Multi-head Attention (MHA) layer. Modified from
#         https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py
#
#         Args:
#             in_len: Length of the input
#             key_len: Length of the key vectors
#             value_len: Length of the value vectors.
#             n_heads: Number of attention heads
#             n_pos_features: Number of positional embedding features
#             pos_dropout: Dropout probability in the positional embeddings
#             attn_dropout: Dropout probability in the output layer
#         """
#         super().__init__()
#
#         # Save params
#         self.in_len = in_len
#         self.key_len = key_len
#         self.value_len = value_len
#         self.n_heads = n_heads
#         self.n_pos_features = n_pos_features
#
#         # Create linear layers
#         self.to_q = nn.Linear(self.in_len, self.key_len * self.n_heads, bias=False)
#         self.to_k = nn.Linear(self.in_len, self.key_len * self.n_heads, bias=False)
#         self.to_v = nn.Linear(self.in_len, self.value_len * self.n_heads, bias=False)
#         self.to_out = nn.Linear(self.value_len * self.n_heads, self.in_len)
#
#         # relative positional encoding
#         self.positional_embed = get_central_mask
#         self.to_pos_k = nn.Linear(
#             self.n_pos_features, self.key_len * self.n_heads, bias=False
#         )
#         self.rel_content_bias = nn.Parameter(
#             torch.randn(1, self.n_heads, 1, self.key_len)
#         )
#         self.rel_pos_bias = nn.Parameter(torch.randn(1, self.n_heads, 1, self.key_len))
#
#         # dropouts
#         self.pos_dropout = nn.Dropout(pos_dropout)
#         self.attn_dropout = nn.Dropout(attn_dropout)
#
#     def _get_pos_k(self, x):
#         positions = self.positional_embed(x, out_channels=self.n_pos_features)
#         positions = self.pos_dropout(positions)
#         pos_k = self.to_pos_k(positions)
#         pos_k = rearrange(pos_k, "n (h d) -> h n d", h=self.n_heads)
#         return pos_k
#
#     def get_attn_scores(self, x, return_v=False):
#         # Q, K, V
#         q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
#
#         # Get content embeddings
#         q, k, v = map(
#             lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v)
#         )
#         q = q / (self.key_len**0.5)
#
#         # Content logits
#         content_logits = einsum(
#             "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
#         )
#
#         # Positional embeddings
#         pos_k = self._get_pos_k(x)
#
#         # Positional logits
#         pos_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, pos_k)
#         pos_logits = relative_shift(pos_logits)
#
#         # Add content and positional embeddings
#         logits = content_logits + pos_logits
#
#         # Softmax
#         attn = logits.softmax(dim=-1)
#
#         if return_v:
#             return self.attn_dropout(attn), v
#         else:
#             return self.attn_dropout(attn)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Forward pass
#
#         Args:
#             x : Input tensor of shape (N, C, L)
#
#         Returns:
#             Output tensor
#         """
#         # Get attention scores
#         attn, v = self.get_attn_scores(x, return_v=True)
#
#         # Output
#         out = einsum("b h i j, b h j d -> b h i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         return self.to_out(out)
