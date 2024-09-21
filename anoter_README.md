CUDA_VISIBLE_DEVICES=5 python decode_TDS.py --load_checkpoint_path artifacts/DNA_value:v0/human_enhancer_diffusion_enformer_7_11_1536_16_ep10_it3500.pt --task dna --alpha 0.5


CUDA_VISIBLE_DEVICES=6 python decode_DPS.py --load_checkpoint_path artifacts/DNA_value:v0/human_enhancer_diffusion_enformer_7_11_1536_16_ep10_it3500.pt --task dna --guidance_scale 100000