import os
import torch


def load_source_latents_t(i_s, t, latents_path, data_type='stage_end'):
    frames = sorted([d for d in os.listdir(latents_path) if os.path.isdir(os.path.join(latents_path, d))])
    
    # latent of all frames in step t
    latents_all = []
    latents_stage_end_all = []
    for frame in frames:
        if data_type != 'stage_end' and not frame.endswith("_reverted_latent_stage_end"):
            latents_t_path = os.path.join(latents_path, f"{frame}/noisy_latents_stage{i_s}_timestep{t+1}.pt")
            print(latents_t_path)
            assert os.path.exists(latents_t_path), f"Missing latents at stage {i_s} t {t} path {latents_t_path}"
            latents = torch.load(latents_t_path).float()
            latents_all.append(latents)
        
        if data_type == 'stage_end' and frame.endswith("_reverted_latent_stage_end"):
            latents_t_path = os.path.join(latents_path, f"{frame}/noisy_latents_stage{i_s}.pt")
            assert os.path.exists(latents_t_path), f"Missing latents at stage {i_s} path {latents_t_path}"
            latents = torch.load(latents_t_path).float()
            latents_stage_end_all.append(latents)

    if data_type != 'stage_end':
        return latents_all
    else:
        return latents_stage_end_all