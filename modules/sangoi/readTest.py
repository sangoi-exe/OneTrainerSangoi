import torch
from safetensors.torch import load_file
from collections import Counter

sdxl_path  = r"C:\Users\lucas\Downloads\sdXL_v10VAEFix.safetensors"
bigg_path  = r"F:\stable-diffusion-webui-forge\models\diffusers\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\snapshots\743c27bd53dfe508a0ade0f50698f99b39d03bec\1open_clip_pytorch_model.bin"

sdxl_state = load_file(sdxl_path)
bigg_state = torch.load(bigg_path, map_location="cpu")

def normalize(k):
    p = "conditioner.embedders.1.model."
    return k[len(p):] if k.startswith(p) else k

hits = Counter()
for k_s, t_s in sdxl_state.items():
    k_b = normalize(k_s)
    if k_b in bigg_state and bigg_state[k_b].shape == t_s.shape:
        hits['texto'] += 1
    elif f"visual.{k_b}" in bigg_state and bigg_state[f"visual.{k_b}"].shape == t_s.shape:
        hits['visao'] += 1

print("Total SDXL:", len(sdxl_state))
print("Total BigG:", len(bigg_state))
print("Matches texto :", hits['texto'])
print("Matches visão :", hits['visao'])
