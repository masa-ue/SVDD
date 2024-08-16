import wandb

run = wandb.init()

artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/DNA_Diffusion:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/RNA_Diffusion:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/DNA_value:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/RNA_Stability_value:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/RNA_MRL_value:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/DNA_evaluation:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/RNA_evaluation:v0")
dir = artifact.download()
artifact = run.use_artifact("fderc_diffusion/bioseq-optimization/RNA_evaluation_saluki:v0")
dir = artifact.download()

wandb.finish()
