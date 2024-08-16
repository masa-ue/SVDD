import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader_gosai
import diffusion_gosai
import utils
import random
import string
import datetime
import uuid
import wandb
omegaconf.OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))+'_'+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
# def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion_gosai.Diffusion(
      config, 
      # tokenizer=tokenizer
      ).to('cuda')
  
  return diffusion_gosai.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    # tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, test_ds):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch seqs.shape', batch['seqs'].shape)
    print(f'tokens:', dataloader_gosai.dna_detokenize(batch['seqs'][0]))
    print('ids:', batch['seqs'][0])
    # first = batch['input_ids'][0, :k]
    # last = batch['input_ids'][0, -k:]
    # print(f'First {k} tokens:', tokenizer.decode(first))
    # print('ids:', first)
    # print(f'Last {k} tokens:', tokenizer.decode(last))
    # print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config)
                                # tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config)
                                # tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  # if config.get('wandb', None) is not None:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(
    config, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger):
  logger.info('Starting Training.')
  wandb_logger = None
  # wandb_settings = wandb.Settings(
  #     base_url='https://genentech.wandb.io'  # Specify your wandb host URL here
  # )
  # if config.get('wandb', None) is not None and not config.debug_mode:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     settings=wandb_settings,
  #     ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(config)
  # _print_batch(train_ds, valid_ds, test_ds)

  model = diffusion_gosai.Diffusion(
    config, 
    # tokenizer=valid_ds.tokenizer
    )

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  # print('Model eval before training...')
  # trainer.validate(model, dataloaders=valid_ds)
  print('Start training...')
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs_gosai',
            config_name='config_gosai')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  # tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    pass
    # generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    pass
    # _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger)


if __name__ == '__main__':
  main()