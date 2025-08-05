import os
import math
from math import sqrt
from utilities.utils import isinstance_str
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_weight(x, y, timestep=None, save_path=None):
    x = x.clone().detach().cpu().numpy()
    y = y.clone().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x, cmap='viridis')
    axes[0].set_title('Reconstructed attention weight')
    axes[0].axis('off')  
    axes[1].imshow(y, cmap='viridis')
    axes[1].set_title('Editing attention weight')
    axes[1].axis('off')  

    plt.tight_layout()
    assert save_path is not None
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{int(timestep)}.jpg"))
    plt.clf()

@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses(orig_features, target_features, config, timestep, groups=32):
    if config["motion_guidance_type"] == "features_diff_dmt":
        return calculate_losses_feature(orig_features, target_features, config, timestep, groups)
    elif config["motion_guidance_type"] == "text_to_obj_activation":
        return calculate_losses_attention(orig_features, target_features, config, timestep, groups)
    else:
        raise NotImplementedError
        
@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses_attention(orig_attn_weights, target_attn_weights, config, timestep, groups=32):
    # orig_attn_weights: t, h, w

    if config["plot_attn_path"]:
        save_path = config["attn_path"]
        plot_attention_weight(orig_attn_weights[0], target_attn_weights[0], timestep, save_path)

    epsilon = 1e-5
    total_loss = 0
    losses = {}
    if config["attention_l2_weight"] > 0:
        # L1 or L2 loss from mask segmentation
        # loss_fn = nn.SmoothL1Loss()
        loss_fn = nn.MSELoss()
        loss_l2 = loss_fn(orig_attn_weights, target_attn_weights)
        losses["attention_mse_loss"] = loss_l2
        total_loss += loss_l2 * config["attention_l2_weight"]
        print(loss_l2)

    if config["attention_dice_weight"] > 0:
        # DICE loss from mask segmentation
        intersection = torch.sum(orig_attn_weights * target_attn_weights)
        union = torch.sum(orig_attn_weights) + torch.sum(orig_attn_weights)
        print(intersection, union)
        dice_coeff = (2. * intersection + epsilon) / (union + epsilon)
        loss_dice = 1 - dice_coeff
        losses["attention_dice_loss"] = loss_dice
        total_loss += loss_dice * config["attention_dice_weight"]

    if config["attention_wass_weight"] > 0:
        loss_wass = energy_based_attention_loss(
            orig_attn_weights, 
            target_attn_weights, 
            epsilon=epsilon,
            sinkhorn_iter=15
        )
        losses["attention_wass_loss"] = loss_wass
        total_loss += loss_wass * config["attention_wass_weight"]
        print(f'Energy-based attention loss: {loss_wass.item()}')

    losses["total_loss"] = total_loss
    
    return losses

def energy_based_attention_loss(attention_weights_x, attention_weights_y, epsilon=1e-5, sinkhorn_iter=20):
    """
    Computes an entropy-regularized Wasserstein distance loss for 2D attention weights.
    Args:
        attention_weights_x (torch.Tensor): 2D attention weights of shape (batch_size, n).
        epsilon (float): Entropy regularization parameter for stability.
        sinkhorn_iter (int): Number of iterations for Sinkhorn-Knopp algorithm.

    Returns:
        torch.Tensor: Computed Wasserstein distance loss with entropy regularization.
    """
    batch_size, n = attention_weights_x.shape
    loss = 0.0

    # For simplicity, we will compute pairwise Wasserstein distance between attention weights
    for i in range(batch_size):
        # Get the pairwise cost matrix based on the squared difference
        P = attention_weights_x[i] + epsilon  # Add epsilon to avoid log(0)
        Q = attention_weights_x[i] + epsilon
        
        # Compute pairwise cost (euclidean distance)
        C = torch.abs(P.unsqueeze(0) - Q.unsqueeze(1))  # Shape: (n, n)
        
        # Initialize dual variables (u, v) for Sinkhorn
        u = torch.ones(n, 1, device=attention_weights_x.device)
        v = torch.ones(1, n, device=attention_weights_x.device)
        
        # Sinkhorn iterations
        for _ in range(sinkhorn_iter):
            u = 1.0 / (C @ v)
            v = 1.0 / (C.transpose(0, 1) @ u)
        
        # Optimal transport plan and the Wasserstein distance
        T = u * C * v
        wasserstein_dist = torch.sum(T * C)

        # Accumulate the loss
        loss += wasserstein_dist.mean()

    return loss

@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses_feature(orig_features, target_features, config, timestep, groups=32):
    orig = orig_features
    target = target_features

    orig = orig.detach()

    total_loss = 0
    losses = {}
    if len(orig) == 1:
        config["features_loss_weight"] = 1
        config["features_diff_loss_weight"] = 0
    if config["features_loss_weight"] > 0:
        if config["global_averaging"]:
            orig = orig.mean(dim=(2, 3), keepdim=True)
            target = target.mean(dim=(2, 3), keepdim=True)
        
        features_loss = compute_feature_loss(orig, target, groups)
        total_loss += config["features_loss_weight"] * features_loss
        losses["features_mse_loss"] = features_loss

    if config["features_diff_loss_weight"] > 0 and len(orig) > 1:
        features_diff_loss = 0
        orig = orig.mean(dim=(2, 3), keepdim=True)  # t d 1 1
        target = target.mean(dim=(2, 3), keepdim=True)

        for i in range(len(orig)):
            orig_anchor = orig[i]
            target_anchor = target[i]
            orig_diffs = orig - orig_anchor  # t d 1 1
            target_diffs = target - target_anchor  # t d 1 1
            t, d, h, w = orig_diffs.shape
            if groups > 0 and (d%groups) == 0:
                orig_diffs = orig_diffs.reshape(t, -1,groups,h,w)
                target_diffs = target_diffs.reshape(t, -1,groups,h,w)
                features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()
        features_diff_loss /= len(orig)

        total_loss += config["features_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss

    losses["total_loss"] = total_loss
    return losses


def compute_feature_loss(orig, target, groups=32):
    features_loss = 0
    for i, (orig_frame, target_frame) in enumerate(zip(orig, target)):
        d, h, w = orig_frame.shape
        if groups > 0 and (d % groups) == 0:
            orig_frame = orig_frame.contiguous().reshape(-1,groups,h,w)
            target_frame = target_frame.contiguous().reshape(-1,groups,h,w)
        features_loss += 1 - F.cosine_similarity(target_frame, orig_frame.detach(), dim=0).mean()
    features_loss /= len(orig)
    return features_loss


def register_time(model, t):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "t", t)

def register_frame_index(model, frame_index):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "frame_index", frame_index)

def register_batch(model, b):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "b", b)

def register_obj_text_start_end_index(model, src_start_index, src_end_index, tgt_start_index, tgt_end_index):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "src_obj_text_start_index", src_start_index)
            setattr(module, "src_obj_text_end_index", src_end_index)
            setattr(module, "tgt_obj_text_start_index", tgt_start_index)
            setattr(module, "tgt_obj_text_end_index", tgt_end_index)

def register_is_src(model, is_src):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "is_src", is_src)

def register_opt_step(model, i):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "opt_step", i)

def register_is_guidance(model, is_guidance):
    for _, module in model.dit.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "is_guidance", is_guidance)

def register_guidance(model):
    guidance_start_timestep = model.guidance_start_timestep
    guidance_stop_timestep = model.guidance_stop_timestep
    num_frames = model.input_frames_latent_ms[0].shape[-3]
    stages = model.stages
    h_ms = [x_.shape[-2] for x_ in  model.input_frames_latent_ms]
    w_ms = [x_.shape[-1] for x_ in  model.input_frames_latent_ms]
    len_text_encoder = 128

    class ModuleWithConvGuidance(torch.nn.Module):
        def __init__(self, module, guidance_start_timestep, guidance_stop_timestep, num_frames, h, w, len_text_encoder, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_start_timestep = guidance_start_timestep
            self.guidance_stop_timestep = guidance_stop_timestep
            self.num_frames = num_frames
            assert module_type in [
                "spatial_convolution",
            ]
            self.module_type = module_type
            if self.module_type == "spatial_convolution":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.len_text_encoder = len_text_encoder
            self.block_name = block_name
            self.config = config
            self.saved_features = None

        def forward(self, input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.module.norm1(hidden_states)
            hidden_states = self.module.nonlinearity(hidden_states)

            if self.module.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.module.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.module.downsample is not None:
                input_tensor = self.module.downsample(input_tensor)
                hidden_states = self.module.downsample(hidden_states)

            hidden_states = self.module.conv1(hidden_states)

            if temb is not None:
                temb = self.module.time_emb_proj(self.module.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.module.norm2(hidden_states)

            if temb is not None and self.module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.module.nonlinearity(hidden_states)

            hidden_states = self.module.dropout(hidden_states)
            hidden_states = self.module.conv2(hidden_states)

            if self.config["guidance_before_res"] and (self.guidance_start_timestep <= self.t <= self.guidance_stop_timestep):
                self.saved_features = rearrange(
                    hidden_states, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            if self.module.conv_shortcut is not None:
                input_tensor = self.module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.module.output_scale_factor

            if not self.config["guidance_before_res"] and (self.guidance_start_timestep <= self.t <= self.guidance_stop_timestep):
                self.saved_features = rearrange(
                    output_tensor, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            return output_tensor

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_start_timestep, guidance_stop_timestep, \
            num_frames, h_ms, w_ms, len_text_encoder, stages, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_start_timestep = guidance_start_timestep
            self.guidance_stop_timestep = guidance_stop_timestep
            self.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
                "spatial_attention",
                "temporal_convolution",
                "upsampler",
                "linear",
            ]
            self.module_type = module_type
            if self.module_type == "temporal_attention":
                self.starting_shape = "(b h w) t d"
            elif self.module_type == "spatial_attention":
                self.starting_shape = "(b t) (h w) d"
            elif self.module_type == "temporal_convolution":
                self.starting_shape = "(b t) d h w"
            elif self.module_type == "upsampler":
                self.starting_shape = "(b t) d h w"
            elif self.module_type == "linear":
                self.starting_shape = "b (t h w) d"
            self.h_ms = h_ms
            self.w_ms = w_ms
            self.len_text_encoder = len_text_encoder
            self.stages = stages
            self.block_name = block_name
            self.config = config        
        
        def get_attention_weights(self, x, num_groups=4):
            batch_size, sequence_length, dimension = x.shape
            group_dim = dimension // num_groups

            x_grouped = x.view(batch_size, sequence_length, group_dim, num_groups)
            x_grouped = x_grouped.permute(0, 3, 1, 2).flatten(0, 1)

            scores = torch.bmm(x_grouped, x_grouped.transpose(1, 2))
            scores = scores / math.sqrt(group_dim)

            scores = scores.view(batch_size, num_groups, sequence_length, sequence_length)
            scores = scores.mean(dim=1)

            return scores
        
        def plot_attention_weights(self, x_in, shape_frames):
            save_path = os.path.join(
                self.config["attn_path"], 
                self.block_name,
                f"frame{self.frame_index}",
                f"timestep{int(self.t)}"
            )
            Path(save_path).mkdir(parents=True, exist_ok=True)

            x = x_in.clone().detach().float()
            len_frames = [self.len_text_encoder] + [shape_[0]*shape_[1] for shape_ in shape_frames] 
            x_frames = torch.split(x, len_frames)

            t2t = x_frames[0].cpu().numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(t2t, cmap='viridis')
            plt.title("Attention Map")
            plt.axis("off")  
            if self.is_src:
                save_path_ = os.path.join(save_path, f"t2v_src.jpg")
            else:
                save_path_ = os.path.join(save_path, f"t2v_tgt_opt{self.opt_step}.jpg")
            plt.savefig(save_path_, bbox_inches="tight", dpi=300)
            plt.clf()

            for idx, (frame, hw) in enumerate(zip(x_frames[1:], shape_frames)):
                if self.is_src:
                    frame = frame[:,self.src_obj_text_start_index:self.src_obj_text_end_index].mean(-1)
                else: 
                    frame = frame[:,self.tgt_obj_text_start_index:self.tgt_obj_text_end_index].mean(-1)
                frame = frame.reshape(hw[0], hw[1]).cpu().numpy()

                plt.figure(figsize=(3, 5))
                plt.imshow(frame, cmap='viridis')
                plt.title("Attention Map")
                plt.axis("off")  

                if self.is_src:
                    save_path_ = os.path.join(save_path, f"past_cond{idx}_src.jpg")
                else:
                    save_path_ = os.path.join(save_path, f"past_cond{idx}_tgt_opt{self.opt_step}.jpg")
                plt.savefig(save_path_, bbox_inches="tight", dpi=300)
                plt.clf()
        
        def plot_attention_weights_all(self, x):
            save_path = os.path.join(
                self.config["attn_path"], 
                self.block_name,
                f"frame{self.frame_index}",
                f"timestep{int(self.t)}"
            )
            Path(save_path).mkdir(parents=True, exist_ok=True)

            x_in = x.clone().detach().float().cpu().numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(x_in, cmap='viridis')
            plt.title("Attention Map")
            plt.axis("off")

            if self.is_src:
                save_path_ = os.path.join(save_path, f"all_src.jpg")
            else:
                save_path_ = os.path.join(save_path, f"all_tgt_opt{self.opt_step}.jpg")
            plt.savefig(save_path_, bbox_inches="tight", dpi=300)
            plt.clf()

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            num_frames = self.num_frames
            if self.module_type == "temporal_attention":
                size = out.shape[0] // self.b
            elif self.module_type == "spatial_attention":
                size = out.shape[1]
            elif self.module_type == "temporal_convolution":
                size = out.shape[2] * out.shape[3]
            elif self.module_type == "upsampler":
                size = out.shape[2] * out.shape[3]
            elif self.module_type == "linear":
                size = out.shape[1] 
                num_frames = 1

            if self.is_guidance and self.guidance_start_timestep <= self.t <= self.guidance_stop_timestep:
                if self.module_type == "linear":
                    size = None

                    len_latent_stages = []
                    shape_latent_stages = []
                    past_frame = min(self.frame_index, len(self.stages)-1)
                    for i_s in range(len(self.stages)):
                        # low_res * past frames
                        len_latent_stage = [
                            self.h_ms[0] * self.w_ms[0] // 4
                            for _ in range(max(self.frame_index - len(self.stages) + 1, 0))
                        ]
                        shape_latent_stage = [
                            [self.h_ms[0]//2, self.w_ms[0]//2] 
                            for _ in range(max(self.frame_index - len(self.stages) + 1, 0))
                        ]
                        len_latent_stage += [self.h_ms[i_s] * self.w_ms[i_s] // 4]
                        shape_latent_stage += [[self.h_ms[i_s]//2, self.w_ms[i_s]//2]]
                        for f_i in range(past_frame):
                            i_s_ = max(i_s-f_i, 0)
                            len_latent_stage += [self.h_ms[i_s_] * self.w_ms[i_s_] // 4]
                            shape_latent_stage.append([self.h_ms[i_s_]//2, self.w_ms[i_s_]//2])
                        len_latent_stages.append(sum(len_latent_stage))  # mmdit [d, h, w] -> [4d, h//2, w//2]
                        shape_latent_stages.append(list(reversed(shape_latent_stage)))

                    for i_s, len_latent_stage in enumerate(len_latent_stages):
                        if (out.shape[1] - self.len_text_encoder) == len_latent_stage:
                            size = self.h_ms[i_s] * self.w_ms[i_s] // 4
                            break
                    assert size is not None

                h, w = int(sqrt(size * self.h_ms[i_s] / self.w_ms[i_s])), int(sqrt(size * self.h_ms[i_s] / self.w_ms[i_s]) * self.w_ms[i_s] / self.h_ms[i_s])
                # last frame in autoregressive model
                if self.module_type == "linear":
                    if self.config["motion_guidance_type"] == "features_diff_dmt":
                        if self.frame_index == 0:
                            self.saved_features = rearrange(
                                out[:, -size:], f"{self.starting_shape} -> b t d h w", t=num_frames, h=h, w=w
                            )
                        else:
                            self.saved_features = rearrange(
                                out[:, -size:] - out[:, -2*size:-size], f"{self.starting_shape} -> b t d h w", t=num_frames, h=h, w=w
                            )
                    elif self.config["motion_guidance_type"] == "text_to_obj_activation":
                        attn_weight = self.get_attention_weights(out)  # b, l, l
                        
                        attn_type = 'all' # 'obj_to_vis'
                        if attn_type == 'obj_to_vis':
                            self.plot_attention_weights(
                                attn_weight[0,:,:self.len_text_encoder], 
                                shape_latent_stages[i_s]
                            )
                            
                            attn_weight = attn_weight.softmax(dim=-1)  # b, l, l
                            attn_weight_v2t = attn_weight[:,:,:self.len_text_encoder]
                            if self.is_src:
                                src_weight = rearrange(
                                    attn_weight_v2t[:, -size:, self.src_obj_text_start_index:self.src_obj_text_end_index], 
                                    'b (h w) l -> b h w l', h=h, w=w,
                                ).mean(-1)
                                self.saved_features = src_weight.unsqueeze(1)  # b, t, h, w
                            else:
                                tgt_weight = rearrange(
                                    attn_weight_v2t[:, -size:, self.tgt_obj_text_start_index:self.tgt_obj_text_end_index], 
                                    'b (h w) l -> b h w l', h=h, w=w,
                                ).mean(-1)
                                self.saved_features = tgt_weight.unsqueeze(1)  # b, t, h, w
                        else:
                            self.plot_attention_weights_all(attn_weight[0])

                            attn_weight = attn_weight.softmax(dim=-1)  # b, l, l
                            self.saved_features = attn_weight[:, -size:].unsqueeze(1) # b, t, hw, l

                else:
                    self.saved_features = rearrange(
                        out, f"{self.starting_shape} -> b t d h w", t=num_frames, h=h, w=w
                    )

            return out

    single_transformer_list = model.config["single_transformer_list"]
    assert len(single_transformer_list) == 1
    for key, indexes in single_transformer_list.items():
        for idx in indexes:
            module = model.dit.single_transformer_blocks[idx]
            # FluxSingleTransformerBlock(
            #   (norm): AdaLayerNormZeroSingle(
            #     (silu): SiLU()
            #     (linear): Linear(in_features=1920, out_features=5760, bias=True)
            #     (norm): LayerNorm((1920,), eps=1e-06, elementwise_affine=False)
            #   )
            #   (proj_mlp): Linear(in_features=1920, out_features=7680, bias=True)
            #   (act_mlp): GELU(approximate='tanh')
            #   (proj_out): Linear(in_features=9600, out_features=1920, bias=True)
            #   (attn): Attention(
            #     (norm_q): RMSNorm()
            #     (norm_k): RMSNorm()
            #     (to_q): Linear(in_features=1920, out_features=1920, bias=True)
            #     (to_k): Linear(in_features=1920, out_features=1920, bias=True)
            #     (to_v): Linear(in_features=1920, out_features=1920, bias=True)
            #   )
            # )
            if model.config["use_proj_out_features"]:
                submodule = module.proj_out
                module.proj_out = ModuleWithGuidance(
                    submodule,
                    guidance_start_timestep, 
                    guidance_stop_timestep,
                    num_frames,
                    h_ms,
                    w_ms,
                    len_text_encoder,
                    stages,
                    block_name=f"FluxSingleTransformerBlock{idx}_pro_out",
                    config=model.config,
                    module_type="linear",
                )
            
            if model.config["use_proj_mlp_features"]:
                submodule = module.proj_mlp
                module.proj_mlp = ModuleWithGuidance(
                    submodule,
                    guidance_start_timestep, 
                    guidance_stop_timestep,
                    num_frames,
                    h_ms,
                    w_ms,
                    len_text_encoder,
                    stages,
                    block_name=f"FluxSingleTransformerBlock{idx}_proj_mlp",
                    config=model.config,
                    module_type="linear",
                )