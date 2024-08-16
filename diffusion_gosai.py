import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader_gosai
import models
import noise_schedule
import utils
import oracle
from scipy.stats import wasserstein_distance, pearsonr

LOG2 = math.log(2)
LOGGER = utils.get_logger(__name__)

# there is some weird bug with wandb if loading the oracle model as a global variable...
ORACLE_MODEL = None # oracle.get_gosai_oracle()


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config):
    # tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    # self.tokenizer = tokenizer
    # self.vocab_size = self.tokenizer.vocab_size
    self.vocab_size = 4
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    # if (not hasattr(self.tokenizer, 'mask_token')
    #     or self.tokenizer.mask_token is None):
    self.mask_index = self.vocab_size
    self.vocab_size += 1
    # else:
    #   self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'cnn':
      self.backbone = models.dnaconv.CNNModel(
        self.config.model, alphabet_size=self.vocab_size, num_cls=3) # num_cls is not used since classifier is always set to False
    elif self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    # elif self.config.backbone == 'dimamba':
    #   self.backbone = models.dimamba.DiMamba(
    #     self.config,
    #     vocab_size=self.vocab_size,
    #     pad_token_id=self.tokenizer.pad_token_id)
    # elif self.config.backbone == 'ar':
    #   self.backbone = models.autoregressive.AR(
    #     self.config,
    #     vocab_size=self.vocab_size,
    #     mask_index=self.mask_index)
    # elif self.config.backbone == 'hf_dit':
    #   self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
    #     self.config.eval.checkpoint_path,
    #     trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    # self.eval_model_tokenizer = transformers.AutoTokenizer.\
    #   from_pretrained(self.gen_ppl_eval_model_name_or_path)
    # if self.eval_model_tokenizer.pad_token is None:
    #   self.eval_model_tokenizer.pad_token =\
    #       self.eval_model_tokenizer.eos_token
    #   self.eval_model_tokenizer.pad_token_id =\
    #       self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

    # subset of data for evaluation
  
    # self.eval_sets_sp = oracle.subset_for_eval(n=config.eval.subset_size) # train_set_sp, valid_set_sp, test_set_sp
    # self.eval_sets_sp_clss = oracle.subset_eval_groundtruth(self.eval_sets_sp) # train_set_sp_clss, valid_set_sp_clss, test_set_sp_clss
    # self.eval_sets_sp_preds = oracle.subset_eval_preds(self.eval_sets_sp, ORACLE_MODEL) # train_preds, valid_preds, test_preds
    # self.eval_sets_sp_kmers = oracle.subset_eval_kmers(self.eval_sets_sp) # train_kmers, valid_kmers, test_kmers
    # print(1)
    # self.emb_pca = oracle.cal_emb_pca(dataloader_gosai.get_datasets_gosai(skip_train=True)[1], n_components=50, oracle_model=ORACLE_MODEL)
    # self.eval_sets_sp_embs_pca = oracle.subset_eval_embs_pca(self.eval_sets_sp, self.emb_pca, ORACLE_MODEL) # train_sp_emb_pca, valid_sp_emb_pca, test_sp_emb_pca
    
  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    
    print('distributed:', distributed)
    # TODO: need to check these two functions
    if distributed:
      sampler_cls = dataloader_gosai.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader_gosai.RandomFaultTolerantSampler
    
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    # print(logits.shape, xt.shape) # [128, 200, 5] [128, 200]
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    # print(unmasked_indices.shape)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    # TODO: where is the sigma configed and input into the model?
    sigma = self._process_sigma(sigma)
    # x = F.one_hot(x, num_classes=self.vocab_size).to(torch.float32)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      # TODO: double check what it's for. looks like it's for text when the sequence is long and need to cut the seq
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['seqs'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    # TODO: where is the returned value used?
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    # return
    # pass
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      # perplexities = []
      all_samples, all_detoeknized_samples = [], []
      # samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        # import time
        # t0 = time.time()
        samples = self._sample().detach().cpu().numpy()

        # print(samples[0])
        # print(samples[1])
        # t1 = time.time()
        # print(f'sample time: {t1 - t0}')
        # print('number of samples: ', len(samples))
        # print(f'tokens:', dataloader_gosai.dna_detokenize(samples[0]))
        # print(f'tokens:', dataloader_gosai.dna_detokenize(samples[1]))
        # 0 and 1 are different, but all four 0s in different gpus are the same.
        # observe the same problem for the original code for text generation, so maybe some random seed issues

        # Decode the samples to be re-tokenized by eval model
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples)
        # t2 = time.time()
        # print(f'detokenize time: {t2 - t1}')
        # text_samples = self.tokenizer.batch_decode(samples)
        # if self.config.eval.compute_generative_perplexity:
        #   perplexities.append(
        #     self.compute_generative_perplexity(
        #       text_samples).cpu())
        all_samples.append(samples)
        all_detoeknized_samples.extend(detokenized_samples)
      all_samples = np.concatenate(all_samples, axis=0)
      # print('all_samples shape:', all_samples.shape)
      # print(len(all_detoeknized_samples))
      # print(all_detoeknized_samples[0])
      # print(all_detoeknized_samples[1])
      # if self.trainer.global_rank == 0 and hasattr(
      #   self.trainer.logger, 'log_table'):
      #   # Log the last generated samples
      #   text_samples = text_samples[
      #     : self.config.sampling.num_sample_log]
      #   self.trainer.logger.log_table(
      #     key=f'samples@global_step{self.global_step}',
      #     columns=['Generated Samples'],
      #     data=[[s] for s in text_samples])
      # if self.config.eval.compute_generative_perplexity:
      #   self.log(name='val/gen_perplexity_incorrect',
      #            value=np.mean(perplexities),
      #            on_epoch=True,
      #            on_step=False,
      #            sync_dist=True)
      #   self.log('val/gen_ppl',
      #            self.gen_ppl_metric,
      #            on_epoch=True,
      #            on_step=False,
      #            sync_dist=True)
      ws_distance_dict = self.cal_wasserstein_distance(all_detoeknized_samples)
      pearsonr_list = self.cal_kmer_pearsonr(all_detoeknized_samples)
      ws_embpca_list = self.cal_ws_distance_embpca(all_detoeknized_samples)
      
      current_step = self.trainer.global_step
      LOGGER.info(f'Current step: {current_step}')
      LOGGER.info(f'Wasserstein distance: {ws_distance_dict}')
      LOGGER.info(f'3mer Pearsonr: {pearsonr_list}')
      LOGGER.info(f'Wasserstein distance embpca: {ws_embpca_list}')

      self.log('val/3mer_pearsonr_train', pearsonr_list[0], on_step=False, on_epoch=True, sync_dist=True)
      self.log('val/3mer_pearsonr_valid', pearsonr_list[1], on_step=False, on_epoch=True, sync_dist=True)
      self.log('val/3mer_pearsonr_test', pearsonr_list[2], on_step=False, on_epoch=True, sync_dist=True)

      self.log('val/ws_embpca_train', ws_embpca_list[0], on_step=False, on_epoch=True, sync_dist=True)
      self.log('val/ws_embpca_valid', ws_embpca_list[1], on_step=False, on_epoch=True, sync_dist=True)
      self.log('val/ws_embpca_test', ws_embpca_list[2], on_step=False, on_epoch=True, sync_dist=True)

      for key in ws_distance_dict:
        for cell_type in ws_distance_dict[key]:
          metric_values = ws_distance_dict[key][cell_type]
          if metric_values:  # Check if the list is not empty
              # Assuming metric_values contains [train_metric, valid_metric, test_metric]
              self.log(f'val/{key}_{cell_type}_train', metric_values[0], on_step=False, on_epoch=True, sync_dist=True)
              self.log(f'val/{key}_{cell_type}_valid', metric_values[1], on_step=False, on_epoch=True, sync_dist=True)
              self.log(f'val/{key}_{cell_type}_test', metric_values[2], on_step=False, on_epoch=True, sync_dist=True)

      # self.log('val/wasserstein_distance', ws_distance_dict,
      #          on_epoch=True, on_step=False, sync_dist=True)
      # self.log('val/kmer_pearsonr', pearsonr_list,
      #           on_epoch=True, on_step=False, sync_dist=True)
      
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))
      
  def cal_wasserstein_distance(self, seqs):
    generated_preds = oracle.cal_gosai_pred(seqs, ORACLE_MODEL)
    # print('generated_preds shape:', generated_preds.shape) # [1024, 3]
    # print(self.eval_sets_sp_clss[0].shape) # [5000, 3]
    # calculate the ws distance between the generated preds and both the ground truth and the preds in train, valid, test, for each cell line
    ws_distance_dict = {'truth': {'hepg2': [], 'k562': [], 'sknsh': []}, 
                        'preds': {'hepg2': [], 'k562': [], 'sknsh': []}} # in the order of train, valid, test in each list
    # self.eval_sets_sp_clss = oracle.subset_eval_groundtruth(self.eval_sets_sp) # train_set_sp_clss, valid_set_sp_clss, test_set_sp_clss
    # self.eval_sets_sp_preds = oracle.subset_eval_preds(self.eval_sets_sp) # train_preds, valid_preds, test_preds
    for set_sp in self.eval_sets_sp_clss:
      ws_distance_dict['truth']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], set_sp[:, 0]))
      ws_distance_dict['truth']['k562'].append(wasserstein_distance(generated_preds[:, 1], set_sp[:, 1]))
      ws_distance_dict['truth']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], set_sp[:, 2]))   
    for set_sp in self.eval_sets_sp_preds:
      ws_distance_dict['preds']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], set_sp[:, 0]))
      ws_distance_dict['preds']['k562'].append(wasserstein_distance(generated_preds[:, 1], set_sp[:, 1]))
      ws_distance_dict['preds']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], set_sp[:, 2])) 
    return ws_distance_dict

  def cal_ws_distance_embpca(self, seqs):
    generated_embs = oracle.cal_gosai_emb(seqs, ORACLE_MODEL)
    generated_embs_pca = self.emb_pca.transform(generated_embs.reshape(generated_embs.shape[0], -1))
    ws_distance_list = []
    for set_sp_emb_pca in self.eval_sets_sp_embs_pca:
      ws_distance_list.append(oracle.get_wasserstein_dist(generated_embs_pca, set_sp_emb_pca))
    return ws_distance_list
  
  def compare_kmer(self, kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]

  def cal_kmer_pearsonr(self, seqs):
    generated_kmer = oracle.count_kmers(seqs)
    pearsonr_list = []
    for set_sp in self.eval_sets_sp_kmers:
      pearsonr_list.append(self.compare_kmer(set_sp, generated_kmer, self.config.eval.subset_size, len(seqs)))
    return pearsonr_list


  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> torch.FloatTensor:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    aaa = self.mask_index 
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x
  
  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5, eval_sp_size=None, cdq=False):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    if cdq:
      all_time_mid_x = [[] for _ in range(num_steps)]
    mid_x = []
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if cdq:
        for j in range(10):
          if self.sampler == 'ddpm':
            x0, x1, x2, x3 = self._ddpm_update_finetune(x, t, dt)
            all_time_mid_x[i].append(torch.clone(x0.detach()))

        x = x0
        if i != num_steps - 1:
          mid_x.append(torch.clone(x.detach()))
      else:
        if self.sampler == 'ddpm':
          x, x1, x2, x3 = self._ddpm_update_finetune(x, t, dt)

        elif self.sampler == 'ddpm_cache':
          p_x0_cache, x_next = self._ddpm_caching_update(
            x, t, dt, p_x0=p_x0_cache)
          if (not torch.allclose(x_next, x)
                  or self.time_conditioning):
            # Disable caching
            p_x0_cache = None
          x = x_next
        else:
          x = self._analytic_update(x, t, dt)
        if i != num_steps - 1:
          mid_x.append(torch.clone(x.detach()))

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)
    if cdq:
      return x.detach(), mid_x, all_time_mid_x
    else:
      return x.detach(), mid_x

  @torch.no_grad()
  def decode_sample(self, num_steps=None, eps=1e-5, eval_sp_size=None, cdq=False):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x, x1, x2, x3 = self._ddpm_update_finetune(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
                or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)

    return x.detach()

  @torch.no_grad()
  def controlled_sample(self, pre_scorer_embedding, pre_scorer_head, num_steps=None, eps=1e-5, eval_sp_size=None, sample_M=10):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x, x1, x2, x3 = self._ddpm_update_finetune_controlled(x, t, dt, pre_scorer_embedding, pre_scorer_head, repeats=sample_M)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)
    return x


  def controlled_sample_DPS(self, pre_scorer_embedding, pre_scorer_head, num_steps=None, eps=1e-5, eval_sp_size=None, sample_M=10):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':  
        x, x1, x2, x3 = self._ddpm_update_finetune_DPS(x, t, dt, pre_scorer_embedding, pre_scorer_head)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)
    return x
  
  @torch.no_grad()
  def controlled_sample_tweedie(self, reward_model, num_steps=None, eps=1e-5, eval_sp_size=None, sample_M=10, options = True, task='dna'):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x, x1, x2, x3 = self._ddpm_update_finetune_controlled_twedie(x, t, dt, reward_model, repeats=sample_M, options = options, task=task)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)
    return x

  @torch.no_grad()
  def _ddpm_update_finetune(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x, x, q_xs, copy_flag

  @torch.no_grad()
  def _ddpm_update_finetune_controlled(self, x, t, dt, pre_scorer_embedding, pre_scorer_head, repeats=10):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    # _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    # return copy_flag * x + (1 - copy_flag) * _x, x, q_xs, copy_flag

    # Generate 10 samples for each position
    samples = [copy_flag * x + (1 - copy_flag) * _sample_categorical(q_xs) for _ in range(repeats)]
    # samples_tensor = torch.stack(samples, dim=0)  # Shape [10, batch_size, seq_length]
    # Get scores for each sample
    scores = []
    for i in range(repeats):
      pre_scorer_input = pre_scorer_embedding(self.transform_samples(samples[i]).float())  # Customize this part based on your pre_scorer input requirements
      scores.append(pre_scorer_head(pre_scorer_input).squeeze())

    # scores = torch.softmax(scores, dim=0)  # Convert scores to probabilities
    # # Sample from the weighted categorical distribution formed by scores
    # final_sample_indices = torch.multinomial(scores, 1).squeeze(0)  # Shape [batch_size, seq_length]
    # # final_samples = samples_tensor[final_sample_indices, torch.arange(x.shape[0]), :]  # Select the chosen samples
    # final_samples = samples[final_sample_indices]
    # # copy_flag = (x != self.mask_index).to(x.dtype)

    # scores = pre_scorer_head(pre_scorer_input).view(x.size(0), 10)  # Reshape scores to [batch_size, 10]
    scores = torch.stack(scores, dim=1)
    scores = torch.softmax(scores, dim=1)  # Convert scores to probabilities for each batch

    # # Sample from the weighted categorical distribution formed by scores
    # final_sample_indices = torch.multinomial(scores, 1).squeeze(1)  # Shape [batch_size]
    # Select the index of the highest score for each batch
    final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Shape [batch_size]
    final_samples = [samples[final_sample_indices[j]][j,:] for j in range(x.size(0))]  # Select the chosen samples using gathered indices
    final_samples = torch.stack(final_samples, dim=0)
    return final_samples, x, q_xs, copy_flag

  def _ddpm_update_finetune_DPS(self, x, t, dt, pre_scorer_embedding, pre_scorer_head):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    x_grad = self.compute_gradient(self.transform_samples(x).float(), pre_scorer_embedding, pre_scorer_head )
    zero_pad = torch.zeros( q_xs.size()[0], q_xs.size()[1], 1).cuda()
    x_grad = torch.cat((x_grad, zero_pad), 2)
    _x = _sample_categorical(q_xs + 1.5 * x_grad)
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x, x, q_xs, copy_flag
  
  def compute_gradient(self, x, pre_scorer_embedding, pre_scorer_head):
    x.requires_grad_(True)
    #pre_scorer_embedding.rnn_layer.train()
    #pre_scorer_head.rnn_layer.train()
    scores = pre_scorer_head(pre_scorer_embedding(x))
    scores = scores.mean()
    scores.backward()
    x_grad = x.grad.clone()

    return x_grad
  
  @torch.no_grad()
  def _ddpm_update_finetune_controlled_twedie(self, x, t, dt, reward_model, repeats=10, options = "True", task="dna"):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    # _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    # return copy_flag * x + (1 - copy_flag) * _x, x, q_xs, copy_flag

    # Generate 10 samples for each position
    samples = [copy_flag * x + (1 - copy_flag) * _sample_categorical(q_xs) for _ in range(repeats)]

    # samples_tensor = torch.stack(samples, dim=0)  # Shape [10, batch_size, seq_length]
    # Get scores for each sample

    #######################################
    #######################################
    #######################################
    #######################################

    scores = []
    for i in range(repeats):
      if options == "True": # Use Tweedie's formula. Aim to calcuate r(E[x_0|x_t])
        expected_x0 = self.forward(samples[i], sigma_s) # Calcualte E[x_0|x_t]
        expected_x0_arg = torch.argmax(expected_x0,dim=2)
        expected_x0_onehot = torch.nn.functional.one_hot(expected_x0_arg)
      else: # Use heuristc to make masked sequnce to be 0. I think you used this one before?
        raw_seq = torch.nn.functional.one_hot(samples[i], 5)
        raw_seq = raw_seq[:,:,0:4]
        copy_flag = (samples[i] != self.mask_index).to(raw_seq.dtype)
        expected_x0_onehot = copy_flag[:, :, None] * raw_seq
      #expected_x0_arg = samples[i] #This means we use raw x_t
      if task == "rna_saluki":
        scorer = reward_model(self.transform_samples_saluki(expected_x0_onehot).float()).detach().squeeze(2)
      else:
        threshold = 1.5
        scorer0 = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 0]
        #scorer1 = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 1]
        #scorer2 = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 2] 
        #reward_pes1 = torch.clamp(5.0*(threshold - scorer1),max=1.0)
        #reward_pes2 = torch.clamp(5.0*(threshold -  scorer2),max=1.0)
        scorer = scorer0 #- 1.0 * ( scorer1 + scorer2 )  ###+  torch.log(torch.clamp(reward_pes1,min= 1e-40) ) + torch.log(torch.clamp(reward_pes2,min= 1e-40) ) 
      scores.append(scorer.squeeze())

      #######################################
      #######################################
      #######################################
      #######################################

    # scores = torch.softmax(scores, dim=0)  # Convert scores to probabilities
    # # Sample from the weighted categorical distribution formed by scores
    # final_sample_indices = torch.multinomial(scores, 1).squeeze(0)  # Shape [batch_size, seq_length]
    # # final_samples = samples_tensor[final_sample_indices, torch.arange(x.shape[0]), :]  # Select the chosen samples
    # final_samples = samples[final_sample_indices]
    # # copy_flag = (x != self.mask_index).to(x.dtype)

    # scores = pre_scorer_head(pre_scorer_input).view(x.size(0), 10)  # Reshape scores to [batch_size, 10]
    scores = torch.stack(scores, dim=1)
    scores = torch.softmax(scores, dim=1)  # Convert scores to probabilities for each batch

    # # Sample from the weighted categorical distribution formed by scores
    # final_sample_indices = torch.multinomial(scores, 1).squeeze(1)  # Shape [batch_size]
    # Select the index of the highest score for each batch
    final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Shape [batch_size]
    final_samples = [samples[final_sample_indices[j]][j,:] for j in range(x.size(0))]  # Select the chosen samples using gathered indices
    final_samples = torch.stack(final_samples, dim=0)
    return final_samples, x, q_xs, copy_flag

  def transform_samples(self, samples, num_classes=4):
    # One-hot encode the tensor but first mask out the '4's
    mask = samples != 4
    valid_samples = samples * mask
    one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)

    # Apply mask to zero out invalid rows
    one_hot_samples = one_hot_samples * mask.unsqueeze(-1)
    return one_hot_samples

  def transform_samples_saluki(self, one_hot_samples, num_classes=4, final_length=12288):
    # Add two zero columns to each sample
    batch_size, seq_len, _ = one_hot_samples.shape
    padding_zeros = torch.zeros(batch_size, seq_len, 2, device=one_hot_samples.device, dtype=one_hot_samples.dtype)
    one_hot_samples = torch.cat((one_hot_samples, padding_zeros), dim=-1)

    saluki_body = np.load('/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/saluki_body_6042.npy')
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
  def _sample_finetune(self, num_steps=None, eps=1e-5, eval_sp_size=None):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    x_list = [ ]
    x_next_list = []
    q_xs_list = []
    mask_list = []

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x_next, x1, x2, x3 = self._ddpm_update_finetune(x, t, dt)
        x_next_list.append(x_next)
        x_list.append(x1)
        q_xs_list.append(x2)
        mask_list.append(x3)
        x = x_next

      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # print(logits.shape) # (batch_size, seq_len, vocab_size)
        # x=argmax of logits of the unmasked tokens
        # no issue with subs; for sedd, if not using [:, :, :-1], some samples will contain the mask token
        x = logits[:, :, :-1].argmax(dim=-1)
    return x_next_list, x_list, q_xs_list, mask_list, x
  

   
  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      # for variance reduction
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      raise NotImplementedError('Sub-sampling not implemented')
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      # else ts are between 0 and 1
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables: # False
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t) # total noise, rate noise
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance) # q(xt|x0)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    # seems that it takes y=x0,xt=M case
    # what is the const term for, seems to be y=M,xt=x0 case and x0 is known so score estimation is precise
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths