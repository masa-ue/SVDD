from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from grelu.lightning import LightningModel
from grelu.sequence.format import convert_input_type
from grelu.utils import make_list
import wandb
import torch
import grelu
import pandas as pd
import grelu.lightning
import grelu.data.dataset
from grelu.resources import artifacts, get_model_by_dataset, get_dataset_by_model
from sklearn.model_selection import train_test_split
from Enformer import BaseModel, OriBaseModel, ConvHead, ConvGRUTrunk


task = 'rna'
run = wandb.init(
        entity='grelu',
        project="RNA-optimization",
        job_type='FA',
        name='train_rna_MRL_oracle',)
# run = wandb.init(entity='grelu', project='human-mpta-sample-2019', job_type='training', name='train')
# artifact = run.use_artifact('dataset:v3')
# dir = artifact.download()
df = pd.read_csv('/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/dataset:v3/dataset.csv.gz', index_col=0)
df_train, df_test = train_test_split(df, test_size=0.2)
df_test, df_val = train_test_split(df_test, test_size=0.5)
print("train data:", df_train.shape)
print("test data:", df_test.shape)
print("val data:", df_val.shape)

model_params = {
    'model_type':'ConvGRUModel',
    'n_tasks': 1,
    'n_conv': 6,
    'stem_channels': 64,
    'channel_init': 64
}

train_params = {
    'task':'regression',
    'loss': 'MSE',
    'lr':1e-4,
    'logger': 'wandb',
    'batch_size': 512,
    'num_workers': 4,
    'devices': [7],
    'save_dir': 'experiment',
    'optimizer': 'adam',
    'max_epochs': 100,
    'checkpoint': True,
}

train_dataset = grelu.data.dataset.DFSeqDataset(df_train)
val_dataset = grelu.data.dataset.DFSeqDataset(df_val)

model = grelu.lightning.LightningModel(model_params=model_params, train_params=train_params)

trainer = model.train_on_dataset(train_dataset, val_dataset)

artifact = wandb.Artifact('model', type='model')

artifact.add_file(trainer.checkpoint_callback.best_model_path, 'model.ckpt')
run.log_artifact(artifact)

test_dataset = grelu.data.dataset.DFSeqDataset(df_test)
model.test_on_dataset(test_dataset, devices=[7], num_workers=4)

# if task == "rna":
#     model = LightningModel.load_from_checkpoint(
#         "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
#         map_location='cpu')
# elif task == "rna_saluki":
#     common_trunk = ConvGRUTrunk(
#         stem_channels=64,
#         stem_kernel_size=15,
#         n_conv=6,
#         channel_init=64,
#         channel_mult=1,
#         kernel_size=5,
#         act_func="relu",
#         conv_norm=True,
#         pool_func=None,  # None, "max", "avg"
#         pool_size=None,
#         residual=True,  # False
#         crop_len=0,
#         n_gru=1,
#         dropout=0.1,  # 0.3
#         gru_norm=True, )
#     human_head = ConvHead(n_tasks=1, in_channels=64, act_func=None, pool_func='avg', norm=False)
#     model = OriBaseModel(embedding=common_trunk, head=human_head)
#     ckpt_human = torch.load(
#         '/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth',
#         map_location='cpu')
#     model.load_state_dict(ckpt_human, strict=True)
# else:
#     raise NotImplementedError
#
# model.cuda()
# model.eval()



wandb.finish()