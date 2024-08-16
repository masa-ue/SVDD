import torch
from torch.utils.data import Dataset
import numpy as np
import re
import json

import anndata
import os
import importlib
import pandas as pd
from torch.utils.data.dataloader import DataLoader


class DNA_reg_Dataset(Dataset):
    def __init__(self, tokenizer, max_len, mode):
        # self.texts = texts
        # self.conditions = conditions  # New addition
        # self.conditions_split_id = conditions_split_id  # New addition
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.fix_condition = fix_condition
        self.mode = mode
        datafile = pd.read_csv("/home/lix361/projects/rna_optimization/generative/artifacts/dataset:v2/dataset.csv.gz")
        # with open('human_enhancer_oracle.txt', 'w') as file:
        #     for i in range(735156):
        #         file.write(datafile['seq'][i] + ' ' + datafile['hepg2'][i] + ' ' + datafile['k562'][i] + ' ' + datafile['sknsh'][i] + '\n')
        if mode == "train":
            self.data = datafile.loc[(datafile['chrom'] != "chr22") & (datafile['chrom'] != "chr21")]
        elif mode == "val":
            self.data = datafile.loc[(datafile['chrom'] == "chr21")]
        elif mode == "test":
            self.data = datafile.loc[(datafile['chrom'] == "chr22")]
        else:
            raise ValueError("mode must be either train or val or test")
        self.data.reset_index(drop=True, inplace=True)

        self.num_labels = 1
        if mode == "train":
            self.norm_mean = self.data['hepg2'].mean()
            self.norm_mean = torch.tensor([float(self.norm_mean)], dtype=torch.float32)
            self.norm_std = self.data['hepg2'].std()
            self.norm_std = torch.tensor([float(self.norm_std)], dtype=torch.float32)

    def __len__(self):
        # return len(self.data)
        if self.mode == "train":
            return len(self.data)*len(self.data['seq'][0])
        else:
            return int(len(self.data)/10 * len(self.data['seq'][0]))

    def __getitem__(self, idx):
        seq_idx = int(idx/200)
        len_idx = int(idx%200)
        text = self.data['seq'][seq_idx][:(len_idx+1)]
        label = self.data['hepg2'][seq_idx]
        # text = self.data['seq'][idx]
        # label = self.data['hepg2'][idx]
        encoded_text = self.tokenizer.batch_encode_plus([text])
        raw_input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int32).squeeze()
        attention_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int32).squeeze()
        # # if self.conditions is not None:
        # #     raw_input_ids = raw_input_ids[1:]  # Remove the first token (<s>)
        # input_ids = raw_input_ids[:-1]
        # targets = raw_input_ids[1:]
        return raw_input_ids, attention_mask, torch.tensor([float(label)], dtype=torch.float32)
        # return {
        #     "input_ids": raw_input_ids,
        #     "attention_mask": attention_mask,
        #     "label": torch.tensor([float(label)], dtype=torch.float)
        # }


class DNA_reg_conv_Dataset(Dataset):
    def __init__(self, mode):
        # self.fix_condition = fix_condition
        self.mode = mode
        datafile = pd.read_csv("/home/lix361/projects/rna_optimization/generative/artifacts/dataset:v2/dataset.csv.gz")
        # with open('human_enhancer_oracle.txt', 'w') as file:
        #     for i in range(735156):
        #         file.write(datafile['seq'][i] + ' ' + datafile['hepg2'][i] + ' ' + datafile['k562'][i] + ' ' + datafile['sknsh'][i] + '\n')
        if mode == "train":
            self.data = datafile.loc[(datafile['chrom'] != "chr22") & (datafile['chrom'] != "chr21")]
        elif mode == "val":
            self.data = datafile.loc[(datafile['chrom'] == "chr21")]
        elif mode == "test":
            self.data = datafile.loc[(datafile['chrom'] == "chr22")]
        else:
            raise ValueError("mode must be either train or val or test")
        self.data.reset_index(drop=True, inplace=True)
        self.seq_len = len(self.data['seq'][0])
        assert self.seq_len == 200, "seq_len is not 200!"
        # self.num_labels = 1
        self.mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.num_features = len(self.mapping)
        if self.mode != "train":
            self.sample_num = int(len(self.data) / 10)
        # if mode == "train":
        #     self.norm_mean = self.data['hepg2'].mean()
        #     self.norm_mean = torch.tensor([float(self.norm_mean)], dtype=torch.float32)
        #     self.norm_std = self.data['hepg2'].std()
        #     self.norm_std = torch.tensor([float(self.norm_std)], dtype=torch.float32)

    def __len__(self):
        # return len(self.data)
        if self.mode == "train":
            return len(self.data)*self.seq_len
        else:
            return int(len(self.data)/10) * self.seq_len

    def __getitem__(self, idx):
        if self.mode == "train":
            seq_idx = int(idx/200)
            len_idx = int(idx%200)
            text = self.data['seq'][seq_idx][:(len_idx+1)]
            label = self.data['hepg2'][seq_idx]
        else:
            len_idx = int(idx / self.sample_num)
            seq_idx = int(idx % self.sample_num)
            text = self.data['seq'][seq_idx][:(len_idx + 1)]
            label = self.data['hepg2'][seq_idx]
        # text = self.data['seq'][idx]
        # label = self.data['hepg2'][idx]
        if len(text) < self.seq_len:
            text_padded = text + "N"*(self.seq_len-len(text))
        else:
            text_padded = text
        onehot_list = []

        # Iterate over each character in the sequence
        for char in text_padded:
            # Initialize a zero vector of length num_features
            one_hot = torch.zeros(self.num_features, dtype=torch.float32)
            # Set the index corresponding to the character to 1
            if char in self.mapping:
                one_hot[self.mapping[char]] = 1
            elif char == 'N':
                pass
            else:
                raise ValueError(f"Character '{char}' not in mapping")
            # Append the one-hot vector to the list
            onehot_list.append(one_hot)

        # Stack the list of one-hot vectors into a tensor
        onehot_tensor = torch.stack(onehot_list)

        if self.mode == "train":
            if len_idx >= 199:
                EOS = 1
            else:
                EOS = 0
            return onehot_tensor, torch.tensor([float(label)], dtype=torch.float32), text, torch.tensor([float(EOS)], dtype=torch.int32)
        else:
            return onehot_tensor, torch.tensor([float(label)], dtype=torch.float32)



class SimpleDNATokenizer:
    def __init__(self, max_length):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.count = 4
        self.max_length = max_length

    def fit_on_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.fit_on_text(line.strip())

    def fit_on_text(self, text):
        for word in text:
            if word not in self.vocab:
                self.vocab[word] = self.count
                self.count += 1

    def encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text]
        sequence = [self.vocab["<s>"]] + sequence + [self.vocab["</s>"]]
        padding_length = self.max_length - len(sequence)

        if padding_length > 0:
            sequence.extend([self.vocab["<pad>"]] * padding_length)

        return sequence[:self.max_length]

    # def decode(self, token_ids):
    #     reverse_vocab = {v: k for k, v in self.vocab.items()}
    #     return ' '.join(reverse_vocab.get(token_id, "<unk>") for token_id in token_ids if
    #                     token_id not in [self.vocab["<pad>"], self.vocab["<s>"], self.vocab["</s>"]])
    def decode(self, token_ids):
        # --- Remove any characters after the <pad> and </s> ---
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        # --- Remove the <s> token ---
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (token_ids == self.vocab["</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"

        # reverse_vocab = {v: k for k, v in self.vocab.items()}
        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        # for token_id in token_ids:
        #     decoded_tokens.append(reverse_vocab.get(token_id, "<unk>"))

        return ''.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text]
        sequence = [self.vocab["<s>"]] + sequence  # Do not add the ending token for generation
        return sequence

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file)

    def token_decode(self, token_id):
        return self.reverse_vocab.get(token_id, "<unk>")

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            self.vocab = json.load(file)
            self.count = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_decoder_func = np.vectorize(self.token_decode)

    def batch_encode_plus(self, texts):
        encodings = self.encode_batch(texts)
        attention_masks = [[float(token != self.vocab["<pad>"]) for token in encoding] for encoding in encodings]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks
        }

