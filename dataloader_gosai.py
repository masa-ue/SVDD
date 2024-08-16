import torch
import pandas as pd
import typing
import math
import utils
import numpy as np
import os

# BASE_DIR = '~/scratch/'
# BASE_DIR = '~/mdlm'
BASE_DIR = '/data/masatoshi/'
LOGGER = utils.get_logger(__name__)
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
# Create an array for fast lookup
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])


def dna_detokenize(seq):
  return ''.join([list(DNA_ALPHABET.keys())[int(i)] for i in seq])

def batch_dna_detokenize(batch_seq):
    """
    batch_seq: numpy array of shape [batch_size, seq_len]
    return: list of strings
    """
    # batch_seq = np.array(batch_seq)
    # Use NumPy's advanced indexing to replace indices with corresponding characters
    detokenized_batch = lookup_array[batch_seq]
    # Join characters in each sequence to form strings
    detokenized_batch = [''.join(seq) for seq in detokenized_batch]
    return detokenized_batch


class DNASequenceDetokenizer:
  def __init__(self):
    # Define the DNA alphabet mapping from nucleotides to indices
    self.dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Create a lookup tensor for fast conversion from indices to nucleotide characters
    # index_to_dna = {v: k for k, v in self.dna_alphabet.items()}
    self.index_to_dna = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    self.unknown_char = 'N'
    # self.lookup_tensor = torch.tensor([index_to_dna[i] for i in range(len(index_to_dna))], dtype=torch.long)

  def detokenize(self, batch_seq):
    """
    Convert a batch of sequences from indices to DNA strings.

    Args:
        batch_seq (torch.Tensor): Tensor of shape [batch_size, seq_len] containing indices of nucleotides.

    Returns:
        list of str: List containing detokenized DNA sequences.
    """
    # Check if the input is a tensor, if not, convert it
    if not isinstance(batch_seq, torch.Tensor):
      batch_seq = torch.tensor(batch_seq, dtype=torch.long)

    batch_seq = batch_seq.numpy()
    detokenized_batch = []
    for seq in batch_seq:
      detokenized_seq = ''.join(self.index_to_dna.get(index, self.unknown_char) for index in seq)
      detokenized_batch.append(detokenized_seq)

    # Map indices to characters using the lookup tensor
    # char_seq = torch.index_select(self.lookup_tensor, 0, batch_seq.view(-1)).view(batch_seq.size())

    # Convert character indices to string list
    # detokenized_batch = [''.join(seq) for seq in char_seq.numpy().astype(str)]

    return detokenized_batch


class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        data_df = pd.read_csv(os.path.join(BASE_DIR, f'gosai_{split}.csv'))
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())
        LOGGER.info(f'Loaded {split} data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}


def get_datasets_gosai(skip_train=False, skip_valid=False):
  if skip_train:
    train_set = None
  else:
    train_set = GosaiDataset(split='train')
  if skip_valid:
    valid_set = None
    test_set = None
  else:
    valid_set = GosaiDataset(split='val')
    test_set = GosaiDataset(split='test')
  return train_set, valid_set, test_set


def get_dataloaders_gosai(config, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  if skip_train:
    train_set = None
  else:
    train_set = GosaiDataset(split='train')

  if skip_valid:
    valid_set = None
    test_set = None
  else:
    valid_set = GosaiDataset(split='val')
    test_set = GosaiDataset(split='test')

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    # train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    # valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader, test_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0

