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

from grelu.resources import artifacts, get_model_by_dataset, get_dataset_by_model
from sklearn.model_selection import train_test_split
from Enformer import BaseModel, OriBaseModel, ConvHead, ConvGRUTrunk

# Directed Evolution
def evolve(
    seqs: Union[List[str], pd.DataFrame],
    model: LightningModel,
    method: str = "ism",
    patterns: Optional[List[str]] = None,
    prediction_transform: Optional[nn.Module] = None,
    seq_transform: Optional[nn.Module] = None,
    max_iter: int = 10,
    positions: List[int] = None,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    genome: Optional[str] = None,
    for_each: bool = True,
    return_seqs: str = "all",
    return_preds: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sequence design by greedy directed evolution

    Args:
        seqs: a set of DNA sequences as strings or genomic intervals
        model: LightningModel object containing a trained deep learning model
        method: Either "ism" or "pattern".
        patterns: A list of subsequences to try inserting into the starting sequence.
        prediction_transform: A module to transform the model output
        seq_transform: A module to asign scores to sequences
        max_iter: Number of iterations
        positions: Positions to mutate. If None, all positions will be mutated
        devices: Device(s) for inference
        num_workers: Number of workers for inference
        batch_size: Batch size for inference
        genome: genome to use if intervals are provided as starting sequences
        for_each: If multiple start sequences are provided, perform directed
            evolution independently from each one
        return_seqs: "all", "best" or "none".
        return_preds: If True, return all the individual model predictions in addition to the
            model prediction score.
        verbose: Print status after each iteration

    Returns:
        A dataframe containing directed evolution results
    """
    from grelu.data.dataset import ISMDataset, MotifScanDataset, SeqDataset

    # Empty dataframe to save outputs
    outputs = pd.DataFrame()

    # Initialize score for best sequence so far
    best_score = -np.Inf

    # Add prediction transform
    model.add_transform(prediction_transform)

    # Give each starting sequence a unique ID
    start_seq_idxs = range(len(seqs))

    # Create starting sequence dataset
    ds = SeqDataset(seqs, genome=genome)

    # Iterate
    for i in range(max_iter + 1):
        if verbose:
            print(f"Iteration {i}")

        # Make dataframe containing current iteration output
        curr_output = pd.DataFrame.from_dict(
            {
                "iter": i,
                "start_seq": start_seq_idxs,
                "best_in_iter": False,
            }
        )

        # Get position and allele
        if i > 0:
            curr_output[["position", "allele"]] = pd.DataFrame(
                [ds.__getitem__(j, return_compressed=True) for j in range(len(ds))],
            )

        # Add prediction score
        curr_output["prediction_score"] = model.predict_on_dataset(
            dataset=ds,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
        ).flatten()

        # Add sequence score
        curr_output["seq_score"] = (
            0
            if seq_transform is None
            else seq_transform(
                [convert_input_type(ds[j], "strings") for j in range(len(ds))]
            )
        )

        # Combine scores
        curr_output["total_score"] = (
            curr_output["prediction_score"] + curr_output["seq_score"]
        )

        # Mark best sequence(s) from current iteration
        curr_best_idxs = (
            curr_output.groupby("start_seq").total_score.idxmax()
            if for_each
            else curr_output.total_score.idxmax()
        )
        curr_output.loc[curr_best_idxs, "best_in_iter"] = True

        # Add full sequences
        if return_seqs == "all":
            curr_output["seq"] = convert_input_type(
                torch.stack([ds[j] for j in range(len(ds))]), "strings"
            )
        elif return_seqs == "best":
            curr_output.loc[curr_best_idxs, "seq"] = convert_input_type(
                torch.stack([ds[j] for j in curr_best_idxs]), "strings"
            )

        # Concatenate outputs
        outputs = pd.concat([outputs, curr_output])

        # Select the best sequences(s) from current iteration
        best = curr_output.loc[
            curr_best_idxs,
            ["seq", "start_seq", "total_score"],
        ]
        overall_best_score = best.total_score.max()
        if verbose:
            # Print the best losses at current iteration
            print(f"Best value at iteration {i}: {overall_best_score:.3f}")

        # Check if best sequence is better than the previous best sequence
        if overall_best_score > best_score:
            best_score = overall_best_score
        else:
            print(f"Score did not increase on iteration: {i}")
            break

        # Mutate sequences for next iteration
        if i < max_iter:
            if method == "ism":
                ds = ISMDataset(make_list(best.seq), positions=positions, drop_ref=True)
            elif method == "pattern":
                ds = MotifScanDataset(
                    make_list(best.seq), motifs=patterns, positions=positions
                )
            else:
                raise NotImplementedError

            start_seq_idxs = make_list(
                best.start_seq.repeat(ds.n_augmented * ds.n_alleles)
            )

    # Remove the transform
    model.reset_transform()

    outputs = outputs.reset_index(drop=True)

    # Get model predictions
    if return_preds:
        ds = SeqDataset(outputs.seq.dropna().tolist())
        preds = model.predict_on_dataset(
            ds,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
        )  # B T L

        # Reshape the predictions for selected tasks
        if prediction_transform is not None:
            preds = preds[:, prediction_transform.tasks, :]
            if prediction_transform.length_aggfunc_numpy is None:
                preds = preds.squeeze(-1)
            else:
                preds = prediction_transform.length_aggfunc_numpy(
                    preds, axis=-1
                )  # B, T

            # Get task names
            task_names = [
                model.data_params["tasks"]["name"][t]
                for t in prediction_transform.tasks
            ]

        else:
            preds = preds.squeeze()
            assert preds.ndim == 1
            task_names = model.data_params["tasks"]["name"]

        # Add model predictions to output dataframe
        outputs[task_names] = np.nan
        outputs.loc[~outputs.seq.isna(), task_names] = preds

    return outputs


# Ledidi
def ledidi(
    seq: str,
    model: Callable,
    prediction_transform: Optional[nn.Module] = None,
    max_iter: int = 20000,
    positions: Optional[List[int]] = None,
    devices: Union[str, int] = "cpu",
    num_workers: int = 1,
    **kwargs,
):
    """
    Sequence design with Ledidi

    Args:
        seq: an initial DNA sequence as a string.
        model: A trained LightningModel object
        prediction_transform: A module to transform the model output
        max_iter: Number of iterations
        positions: Positions to mutate. If None, all positions will be mutated
        targets: List of targets for each loss function
        devices: Index of device to use for inference
        num_workers: Number of workers for inference
        **kwargs: Other arguments to pass on to Ledidi

    Returns:
        Output DNA sequence(s) as strings.

    """
    from ledidi import Ledidi

    # Add the prediction transform
    model.add_transform(prediction_transform)

    def loss_func(x, target):
        return -Tensor(x).mean()

    # Convert sequence into a one-hot encoded tensor
    X = convert_input_type(seq, "one_hot")
    X = X.unsqueeze(0).to(torch.device(devices))

    # Create input mask
    if positions is not None:
        input_mask = Tensor([True] * X.shape[-1]).type(torch.bool)
        input_mask[positions] = False
    else:
        input_mask = None

    # Move model to device
    orig_device = model.device
    model = model.to(torch.device(devices))

    # Initialize ledidi
    designer = Ledidi(
        model,
        X[0].shape,
        output_loss=loss_func,
        max_iter=max_iter,
        input_mask=input_mask,
        target=None,
        **kwargs,
    )
    designer = designer.to(torch.device(devices))

    # Run ledidi
    X_hat = designer.fit_transform(X, None).cpu()

    # Transfer device
    model = model.to(orig_device)

    # Remove the transform
    model.reset_transform()

    # Return sequences as strings
    return convert_input_type(X_hat, "strings")


task = 'rna'
wandb.init(
        entity='grelu',
        project="RNA-optimization",
        job_type='FA',
        name='directed-evolution-baseline')
# run = wandb.init(entity='grelu', project='human-mpta-sample-2019', job_type='training', name='train')
# artifact = run.use_artifact('dataset:v3')
# dir = artifact.download()
df = pd.read_csv('/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/dataset:v3/dataset.csv.gz', index_col=0)
df_train, df_test = train_test_split(df, test_size=0.2)
train_dataset = grelu.data.dataset.DFSeqDataset(df_train)

if task == "rna":
    model = LightningModel.load_from_checkpoint(
        "/home/lix361/projects/rna_optimization/controlled_decoding_diffusion/artifacts/model:v8/model.ckpt",
        map_location='cpu')
elif task == "rna_saluki":
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
    model = OriBaseModel(embedding=common_trunk, head=human_head)
    ckpt_human = torch.load(
        '/home/lix361/projects/rna_optimization/prediction_half_life/storage/ConvGRUModel_nochange_nopool_residual_ConvHeadnoactnonorm_dp0.1_lr1e-4_noclip_interbatch/epoch31/model_human.pth',
        map_location='cpu')
    model.load_state_dict(ckpt_human, strict=True)
else:
    raise NotImplementedError

model.cuda()
model.eval()

labels_flattened = train_dataset.labels.reshape(-1)
B = 64

# Get the indices of the top B labels
top_B_indices = np.argsort(labels_flattened)[-B:]

# Select the sequences corresponding to the top B labels
top_B_seqs = train_dataset.seqs[top_B_indices]

# Mapping from numeric values to nucleotide bases
int_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# Convert each sequence to a string
seqs_as_strings = [''.join(int_to_base[num] for num in seq) for seq in top_B_seqs]

output = evolve(
    seqs_as_strings,  # The initial sequences
    model,
    max_iter=7,  # Number of iterations for directed evolution
    num_workers=8,
    devices=0,
    return_seqs="all",  # Return all the evolved sequences
)

import grelu.visualize
grelu.visualize.plot_evolution(output, outlier_size=.1).save('plot_evol.png')
wandb.log({"plot_evol": wandb.Image('plot_evol.png')})

end_seq = output[output.iter==7].sort_values('total_score').iloc[-1].seq
idx = output[output.iter==7].sort_values('total_score').iloc[-1].start_seq.item()
start_seq = seqs_as_strings[idx]
print("Start_seq", start_seq)
print("Best end seq: ", end_seq)

import grelu.sequence.mutate
mutated_positions = grelu.sequence.mutate.seq_differences(start_seq, end_seq, verbose=True)

import grelu.interpret.score

start_attrs = grelu.interpret.score.get_attributions(
    model, start_seq, device=0,
    method="inputxgradient",
)
end_attrs = grelu.interpret.score.get_attributions(
    model, end_seq, device=0,
    method="inputxgradient",
)

grelu.visualize.plot_attributions(
    start_attrs,
    highlight_positions=mutated_positions, # Highlight the mutated positions
    ticks=10,
).ax.figure.savefig('start_seq_logo.png')
wandb.log({"start_seq_logo": wandb.Image('start_seq_logo.png')})

grelu.visualize.plot_attributions(end_attrs, highlight_positions=mutated_positions, ticks=10,).ax.figure.savefig('end_seq_logo.png')
wandb.log({"end_seq_logo": wandb.Image('end_seq_logo.png')})

wandb.finish()