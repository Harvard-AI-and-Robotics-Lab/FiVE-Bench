import argparse
import copy
import os, math, cv2
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import logging, T5TokenizerFast
from torchvision import transforms
from diffusers.utils import export_to_video
from typing import List, Union

from torchvision.transforms.functional import InterpolationMode

from utilities.guidance_utils import register_batch
from pyramid_dit import PyramidDiTForVideoGeneration

# suppress partial model loading warning
logging.set_verbosity_error()


class T5Tokenizer(torch.nn.Module):
    def __init__(self, model_name, model_path):
        super().__init__()
        if model_name == "pyramid_flux":
            self.tokenizer = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_2'))
        elif model_name == "pyramid_mmdit":
            self.tokenizer = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer_3'))
        else:
            raise NotImplementedError(f"Unsupported Text Encoder")
    
    def forward(
        self, 
        prompt: Union[str, List[str]] = None,
        obj_prompt: Union[str, List[str]] = None,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        print('Prompt len:', len(text_input_ids), text_input_ids)

        # Tokenize the object phrase
        obj_prompt = [obj_prompt] if isinstance(obj_prompt, str) else obj_prompt
        obj_inputs = self.tokenizer(
            obj_prompt,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        obj_input_ids = obj_inputs.input_ids[0]
        obj_input_ids = obj_input_ids[:-1]  # Remove start/end tokens
        print('Obj prompt len:',len(obj_input_ids), obj_input_ids)

        # Find the start index of the phrase in the sentence
        start_idx = -1
        for i in range(len(text_input_ids) - len(obj_input_ids) + 1):
            if text_input_ids[i:i+len(obj_input_ids)].tolist() == obj_input_ids.tolist():
                start_idx = i
                break
        
        # Output results
        # assert start_idx != -1, "Phrase not found in sentence tokens."
        if start_idx == -1:
            print("Phrase not found in sentence tokens.")  # Not used
        end_idx = start_idx + len(obj_input_ids)

        return start_idx, end_idx
    

class VideoFrameProcessor:
    # load a video and transform
    def __init__(self, resolution=384, num_frames=41, add_normalize=True, sample_fps=24):
    
        image_size = resolution

        transform_list = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
        ]
        
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        print(f"Transform List is {transform_list}")
        self.num_frames = num_frames
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps

            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]

            if len(frames) < self.num_frames:
                num_frame_to_pack = self.num_frames - len(frames)
                recurrent_num = num_frame_to_pack // len(frames)
                frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
                assert len(frames) >= self.num_frames, f'{len(frames)}'

            frames = torch.stack(frames).float() / 255
            frames = self.transform(frames)
            frames = frames.permute(1, 0, 2, 3)

            return frames, None
            
        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None, None


class Guidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        model_dtype = config["dtype"]
        assert model_dtype == "bf16", "Pyramid-Flow performs better for bf16!!"
        if model_dtype == "bf16":
            # inference only, "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
            torch_dtype = torch.bfloat16 
        elif model_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.dtype = torch_dtype  

        self.guidance_start_timestep = config["guidance_start_timestep"]
        self.guidance_stop_timestep = config["guidance_stop_timestep"]
        self.guidance_start_timestep_first = config["guidance_start_timestep_first"]
        self.guidance_stop_timestep_first = config["guidance_stop_timestep_first"]

        if config['resolution'] == '384p':
            self.resolution = (640, 384)  # width, height
        elif config['resolution'] == '768p':
            self.resolution = (1280, 768)
        else:
            raise ValueError
        self.ori_resolution = None
        
        print("\n\nLoading video model ...")

        model_name = config["model_name"]  # "pyramid_flux" or "pyramid_mmdit"
        if config['resolution'] == '384p':
            variant='diffusion_transformer_384p'           # For low resolution
        else:
            variant='diffusion_transformer_768p'           # For high resolution
        model_path = config["model_path"]   # The downloaded checkpoint dir

        self.t5_tokenizer = T5Tokenizer(model_name, model_path)

        self.video_pipe = PyramidDiTForVideoGeneration(
            model_path,
            model_dtype=self.dtype,
            model_name=model_name,
            model_variant=variant,
        )

        self.video_pipe.vae.enable_tiling()
        self.video_pipe._guidance_scale = config["guidance_scale"]
        self.vae = self.video_pipe.vae.to("cuda").to(self.dtype)
        self.text_encoder = self.video_pipe.text_encoder.to("cuda")
        self.dit = self.video_pipe.dit.to("cuda")
        self.decode_latent = self.video_pipe.decode_latent
        self.scheduler = copy.deepcopy(self.video_pipe.scheduler)
        self.stages = self.video_pipe.stages
        self.do_classifier_free_guidance = self.video_pipe.do_classifier_free_guidance
        self.device = self.video_pipe.device
        print("video model loaded!\n\n")

        self.generator = None

        with torch.no_grad():
            # T5 text embed, T5 text mask, CLIP text pooled embed
            self.src_text_prompt_cond, self.src_prompt_attention_mask, self.src_pooled_prompt_embeds, self.src_all_prompt_embeds = self.get_text_embeds(
                config["source_prompt"], config["negative_prompt"], 
            )
            self.tgt_text_prompt_cond, self.tgt_prompt_attention_mask, self.tgt_pooled_prompt_embeds, self.tgt_all_prompt_embeds = self.get_text_embeds(
                config["target_prompt"], config["negative_prompt"], 
            )

        self.video_processor = VideoFrameProcessor(
            (self.resolution[1], self.resolution[0]), num_frames=self.config["max_frames"], add_normalize=True
        )   

        # load images and latents
        self.frame_index = None
        self.input_frames_latent_ms, self.noise_latent_ms = self.get_data()

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, cpu_offloading=False):
        if isinstance(prompt, str):
            if len(prompt) > 0:  # except null prompt
                prompt = prompt + ", hyper quality, Ultra HD, 8K"   # adding this prompt to improve aesthetics
        else:
            assert isinstance(prompt, list)
            prompt = [p_ + ", hyper quality, Ultra HD, 8K" if len(p_) > 0 else p_ for p_ in prompt]
        
        negative_prompt = negative_prompt or ""

        # Get the text embeddings
        if cpu_offloading:
            self.text_encoder.to("cuda")
        prompt_embeds, prompt_attention_mask, pooled_prompt_embeds, all_prompt_embeds = self.text_encoder(
            prompt, self.device, return_all_prompt_embeds_clip=True)
        negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds, negative_all_prompt_embeds = self.text_encoder(
            negative_prompt, self.device, return_all_prompt_embeds_clip=True)
        
        if cpu_offloading:
            self.text_encoder.to("cpu")
            self.vae.to("cuda")
            torch.cuda.empty_cache()
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            all_prompt_embeds = torch.cat([negative_all_prompt_embeds, all_prompt_embeds], dim=0)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds, all_prompt_embeds


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def get_data(self):
        # load video frames
        data_path = self.config["data_path"]

        if os.path.isdir(data_path):
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem))
            if len(images) > self.config["max_frames"]:
                print('!'*100)
                print(f'Video frames {len(images)} > Max frames {self.config["max_frames"]}! Use the first {self.config["max_frames"]} frames.')
                print('!'*100)
                images = images[:self.config["max_frames"]]
            width, height = Image.open(images[0]).size
            self.ori_resolution = (height, width)

            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
            input_frames_tensor_list = []
            for unit_index in tqdm(range(len(images))):
                image_name = images[unit_index]
                image = Image.open(image_name).convert("RGB")
                image = image.resize(self.resolution)
                input_image_tensor = image_transform(image).unsqueeze(0).unsqueeze(2)   # [b c 1 h w]
                input_frames_tensor_list.append(input_image_tensor)

            input_frames_latent = torch.cat(input_frames_tensor_list, dim=2)
        
        else:

            input_frames_latent, _ = self.video_processor(data_path)
            input_frames_latent = input_frames_latent.unsqueeze(0)

            self.ori_resolution = (input_frames_latent.shape[-2], input_frames_latent.shape[-1])

        # 8n + 1
        nf = input_frames_latent.shape[2] // 8
        nf = 8*nf+1 if input_frames_latent.shape[2] % 8 != 0 else 8*(nf-1)+1
        input_frames_latent = input_frames_latent[:,:,:nf]

        input_frames_latent = self.vae.encode(input_frames_latent.to(self.device).to(self.dtype)).latent_dist.sample() 

        input_frames_latent[:,:,:1] = (input_frames_latent[:,:,:1] - self.video_pipe.vae_shift_factor) * self.video_pipe.vae_scale_factor  # [b c 1 h w]
        input_frames_latent[:,:,1:] = (input_frames_latent[:,:,1:] - self.video_pipe.vae_video_shift_factor) * self.video_pipe.vae_video_scale_factor  # [b c 1 h w]
        input_frames_latent_ms = self.video_pipe.get_pyramid_latent(input_frames_latent, len(self.stages) - 1)

        # prepare noisy latent
        if self.config["seed"] is None:
            if os.path.exists(os.path.join(self.config["latents_path"], "seed.txt")):
                with open(os.path.join(self.config["latents_path"], "seed.txt"), "r") as file:
                    seed = file.read().strip()  # Remove any surrounding whitespace or newline characters
                seed = int(seed)
            else:
                seed = torch.randint(0, 1000000, (1,)).item()
            self.config["seed"] = seed
        else:
            seed = self.config["seed"]
        Path(self.config["output_path"]).mkdir(exist_ok=True)
        with open(Path(self.config["output_path"], "seed.txt"), "w") as f:
            f.write(str(seed))
            
        self.generator = torch.Generator()
        self.generator.manual_seed(self.config["seed"])

        # Create the initial random noise
        batch_size, num_channels_latents = input_frames_latent.shape[:2]
        noise_latent = self.video_pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            input_frames_latent.shape[2],
            self.resolution[1],   # height bfe VAE Enc
            self.resolution[0],   # width bfe VAE Enc
            self.dtype,
            self.device,
            generator=self.generator,
        )
        noise_latent = noise_latent[:,:,:1].expand(noise_latent.shape)
        height, width = noise_latent.shape[-2:]
        noise_latent_ms = [noise_latent.clone()]
        # by defalut, we needs to start from the block noise
        for _ in range(1, len(self.stages)):
            height //= 2; width //= 2
            noise_latent = rearrange(noise_latent, 'b c t h w -> (b t) c h w')
            noise_latent = F.interpolate(noise_latent, size=(height, width), mode='bilinear') * 2
            noise_latent = rearrange(noise_latent, '(b t) c h w -> b c t h w', b=batch_size)
            noise_latent_ms.append(noise_latent)
        noise_latent_ms = list(reversed(noise_latent_ms))   # make sure from low res to high res

        return (
            input_frames_latent_ms,
            noise_latent_ms,
        )
        
    @torch.no_grad()
    def get_sk_ek_sigma(self, i_s, allocation_type="latent-enhanced"): 
        timesteps = self.scheduler.timesteps
        s_k = timesteps[0] / self.scheduler.config.num_train_timesteps
        e_k = timesteps[-1] / self.scheduler.config.num_train_timesteps

        if allocation_type == "latent-enhanced":
            # elf.scheduler.start_sigmas: {0: 1.0, 1: 0.8002399489209289, 2: 0.5007496155411024}
            # noise precent s_k, e_k: S0 [1, 0.5], S1 [0.67, 0.2], S2 [0.33, 0]
            s_k_sigma = s_k
            e_k_sigma = 1 - self.scheduler.start_sigmas[len(self.stages)-1-i_s]
        elif allocation_type == "equal":
            # s_k, e_k: S0 [1, 0.667], S1[0.667, 0.334], S2 [0.334, 0]
            s_k_sigma = torch.tensor(1 - i_s / len(self.stages)).to(s_k)
            e_k_sigma = torch.tensor(1 - (i_s+1) / len(self.stages)).to(s_k)
        elif allocation_type == "timesteps":
            # s_k, e_k: S0 [1, 0.74], S1[0.74, 0.38], S2 [0.38, 0]
            s_k_sigma, e_k_sigma = s_k, e_k 
        else:
            assert ValueError

        return s_k_sigma, e_k_sigma

    @torch.no_grad()
    def denoise_step(self, i_s, i, t, past_condition_latent_src, past_condition_latent_tgt, 
                     y_0_s_k_src, y_0_s_k_tgt, y_0_e_k_src, y_0_e_k_tgt):
        register_batch(self, 4)
        # interpolate the current latent in timestep t
        s_k = self.scheduler.timesteps[0] / self.scheduler.config.num_train_timesteps
        e_k = self.scheduler.timesteps[-1] / self.scheduler.config.num_train_timesteps
        t_01 = t / self.scheduler.config.num_train_timesteps
        t_ = (t_01 - e_k) / (s_k - e_k)  # t_ -> 0

        x_src = t_ * y_0_s_k_src + (1 - t_) * y_0_e_k_src
        x_tgt = y_0_e_k_tgt + x_src - y_0_e_k_src   # FlowEdit

        latent_model_input = torch.cat(
            [x_src] * 2 + [x_tgt] * 2
        ) if self.do_classifier_free_guidance else torch.cat([x_src, x_tgt])
        
        latent_model_input = [
            torch.cat([p_src, p_tgt])
            for p_src, p_tgt in zip(past_condition_latent_src[i_s], past_condition_latent_tgt[i_s])
        ] + [latent_model_input]

        text_prompt_cond = torch.cat([self.src_text_prompt_cond, self.tgt_text_prompt_cond])
        prompt_attention_mask = torch.cat([self.src_prompt_attention_mask, self.tgt_prompt_attention_mask])
        pooled_prompt_embeds = torch.cat([self.src_pooled_prompt_embeds, self.tgt_pooled_prompt_embeds])

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input[-1].shape[0]).to(x_src.dtype).to(x_src.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise_pred = self.dit(
                sample=[latent_model_input],
                timestep_ratio=timestep,
                encoder_hidden_states=text_prompt_cond,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
            )[0]

            noise_pred_uncond_src, noise_pred_cond_src, noise_pred_uncond_tgt, noise_pred_cond_tgt = noise_pred.chunk(4)
            
            if self.frame_index == 0:
                tgt_guidance_scale = 10.0 + i_s * 2
                noise_pred_src = noise_pred_uncond_src + self.config["guidance_scale"] * (noise_pred_cond_src - noise_pred_uncond_src)
                noise_pred_tgt = noise_pred_uncond_tgt + tgt_guidance_scale * (noise_pred_cond_tgt - noise_pred_uncond_tgt)
            else:
                tgt_guidance_scale = 10.0 + i_s * 2
                noise_pred_src = noise_pred_uncond_src + self.config["video_guidance_scale"] * (noise_pred_cond_src - noise_pred_uncond_src)
                noise_pred_tgt = noise_pred_uncond_tgt + tgt_guidance_scale * (noise_pred_cond_tgt - noise_pred_uncond_tgt)
                
        noise_pred_diff = noise_pred_tgt - noise_pred_src

        self.scheduler._step_index = i
        y_0_e_k_tgt = self.scheduler.step(
            model_output=noise_pred_diff,
            timestep=timestep,
            sample=y_0_e_k_tgt,
            generator=self.generator,
        ).prev_sample       

        return y_0_e_k_tgt
    
    @torch.no_grad()
    def sample_block_noise(self, bs, ch, temp, height, width):
        gamma = self.scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(4), 
            torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma
        )
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',
                          b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise
    
    def upsample_with_jump_points(self, i_s, latents_src, latents_tgt, return_latents_bfe_block_noise=False):
        temp = latents_tgt.shape[2]
        height = latents_tgt.shape[-2] * 2
        width = latents_tgt.shape[-1] * 2
        latents_src = rearrange(latents_src, 'b c t h w -> (b t) c h w')
        latents_src = F.interpolate(latents_src, size=(height, width), mode='nearest')
        latents_src = rearrange(latents_src, '(b t) c h w -> b c t h w', t=temp)
        latents_tgt = rearrange(latents_tgt, 'b c t h w -> (b t) c h w')
        latents_tgt = F.interpolate(latents_tgt, size=(height, width), mode='nearest')
        latents_tgt = rearrange(latents_tgt, '(b t) c h w -> b c t h w', t=temp)

        latents_src_clone, latents_tgt_clone = latents_src.clone(), latents_tgt.clone()

        # Fix the stage, ori_start_sigmas: {0: 1.0, 1: 0.6669999957084656, 2: 0.33399999141693115}
        # stage 1: alpha=0.599, beta=0.693   => alpha: mean shift, beta: conv shift
        # stage 2: alpha=0.749, beta=0.433
        ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
        gamma = self.scheduler.config.gamma  # 0.333
        alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
        beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
        
        # add noise per block
        bs, ch, temp, height, width = latents_tgt.shape
        noise = self.sample_block_noise(bs, ch, temp, height, width)
        noise = noise.to(device=self.device, dtype=self.dtype)
        latents_src = alpha * latents_src + beta * noise    # To fix the block artifact
        latents_tgt = alpha * latents_tgt + beta * noise    # To fix the block artifact

        if return_latents_bfe_block_noise:
            return latents_src, latents_tgt, latents_src_clone, latents_tgt_clone
        return latents_src, latents_tgt
    
    @torch.no_grad()
    def get_past_condition_latents(self, src_latent_list, tgt_latent_list):
        batch_size = self.input_frames_latent_ms[0].shape[0]
        is_first_frame = self.frame_index == 0

        if is_first_frame:
            past_condition_latent_src = [[] for _ in range(len(self.stages))]
            past_condition_latent_tgt = [[] for _ in range(len(self.stages))]
        else:
            past_condition_latent_src = []
            clean_latents_list_pyramid = [x[:,:,:self.frame_index] for x in self.input_frames_latent_ms]
            
            use_corrupt_noise = False
            for i_s in range(len(self.stages)):
                last_cond_latent = clean_latents_list_pyramid[i_s][:,:,-1:]
                if use_corrupt_noise:
                    last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=self.device) * self.video_pipe.corrupt_ratio
                    while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                        last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)
                    # We adding some noise to corrupt the clean condition
                    last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent

                stage_input = [torch.cat([last_cond_latent] * 2) if self.video_pipe.do_classifier_free_guidance else last_cond_latent]
        
                # pad the past clean latents
                cur_unit_num = self.frame_index
                cur_stage = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list_pyramid[cur_stage][:, :, -cur_unit_ptx : -(cur_unit_ptx - 1)]
                    if use_corrupt_noise:
                        # We adding some noise to corrupt the clean condition
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = clean_latents_list_pyramid[0][:, :, :-cur_unit_ptx]
                    if use_corrupt_noise:
                        # We adding some noise to corrupt the clean condition
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
            
                stage_input = list(reversed(stage_input))
                past_condition_latent_src.append(stage_input)
                
            past_condition_latent_tgt = []
            reconstructed_latents_list_pyramid = self.video_pipe.get_pyramid_latent(torch.cat(tgt_latent_list, dim=2), len(self.stages) - 1)
            for i_s in range(len(self.stages)):
                last_cond_latent = reconstructed_latents_list_pyramid[i_s][:,:,-1:]
                if use_corrupt_noise:
                    last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=self.device) * self.video_pipe.corrupt_ratio
                    while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                        last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)
                    # We adding some noise to corrupt the clean condition
                    last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent

                stage_input_tgt = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
        
                # pad the past clean latents
                cur_unit_num = self.frame_index
                cur_stage = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = reconstructed_latents_list_pyramid[cur_stage][:, :, -cur_unit_ptx : -(cur_unit_ptx - 1)]
                    if use_corrupt_noise:
                        # We adding some noise to corrupt the clean condition
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input_tgt.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = reconstructed_latents_list_pyramid[0][:, :, :-cur_unit_ptx]
                    if use_corrupt_noise:
                        # We adding some noise to corrupt the clean condition
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents                        
                    stage_input_tgt.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
            
                stage_input_tgt = list(reversed(stage_input_tgt))
                past_condition_latent_tgt.append(stage_input_tgt)
            
        return past_condition_latent_src, past_condition_latent_tgt

    def run_per_unit(self, past_condition_latent_src, past_condition_latent_tgt):
        print("-"*30 + f"frame {self.frame_index} editing" + "-"*30)
        start_timestep_ = self.guidance_start_timestep_first if self.frame_index == 0 \
            else self.guidance_start_timestep
        stop_timestep_ = self.guidance_stop_timestep_first if self.frame_index == 0 \
            else self.guidance_stop_timestep
        print(f"Frame index: {self.frame_index}, Start_timestep: {start_timestep_}, End_timestep: {stop_timestep_}")
        
        y_0_e_k_src_ms = [[] for _ in range(len(self.stages))]
        y_0_e_k_tgt_ms = [[] for _ in range(len(self.stages))]

        for i_s in range(len(self.stages)):

            self.scheduler.set_timesteps(self.n_timesteps, i_s,  device="cuda")
            timesteps = self.scheduler.timesteps
            
            s_k_sigma, e_k_sigma = self.get_sk_ek_sigma(i_s)  # 0.5/0.2/0.0
            frame_latent = self.input_frames_latent_ms[i_s][:,:,[self.frame_index]].clone()
            noise_latent = self.noise_latent_ms[i_s][:,:,[self.frame_index]].clone()
            y_0_e_k_src = (1 - e_k_sigma) * frame_latent + e_k_sigma * noise_latent.clone()  
            
            if i_s == 0:
                y_0_s_k_src = noise_latent.clone()
                y_0_s_k_tgt = noise_latent.clone()
                y_0_e_k_tgt = y_0_e_k_src.clone().detach()
                
            else:
                y_0_s_k_src = y_0_e_k_src_ms[i_s-1][-1].clone().detach()
                y_0_s_k_tgt = y_0_e_k_tgt_ms[i_s-1][-1].clone().detach()
                y_0_s_k_src, y_0_s_k_tgt, _, _ = self.upsample_with_jump_points(
                    i_s, y_0_s_k_src, y_0_s_k_tgt, return_latents_bfe_block_noise=True
                )
                # add the noise diff of src video between start and end points to tgt video
                # (y_0_e_k_src - y_0_s_k_src) contains the info of the source video to be removed
                y_0_e_k_src = y_0_s_k_src + (y_0_e_k_src - y_0_s_k_src)
                y_0_e_k_tgt = y_0_s_k_tgt + (y_0_e_k_src - y_0_s_k_src)
                
            for i in tqdm(range(len(timesteps)), desc="Sampling"):
                t = timesteps[i]

                if not stop_timestep_ <= t <= start_timestep_:
                    continue

                y_0_e_k_tgt = self.denoise_step(
                    i_s, i, t, 
                    past_condition_latent_src, past_condition_latent_tgt,
                    y_0_s_k_src, y_0_s_k_tgt, y_0_e_k_src, y_0_e_k_tgt,
                )
                
            y_0_e_k_src_ms[i_s].append(y_0_e_k_src.clone().detach())
            y_0_e_k_tgt_ms[i_s].append(y_0_e_k_tgt.clone().detach())

        return y_0_e_k_src, y_0_e_k_tgt


    def run(self):
        src_latent_list, tgt_latent_list = [], []

        temp = self.input_frames_latent_ms[0].shape[2]
        for unit_index in tqdm(range(temp)):
            self.frame_index = unit_index
            self.n_timesteps = config["n_timesteps"] if unit_index == 0 else config["n_timesteps"] // 2

            past_condition_latent_src, past_condition_latent_tgt = \
                self.get_past_condition_latents(
                    src_latent_list,
                    tgt_latent_list
                )

            # sampling process
            src_latent, tgt_latent = self.run_per_unit(
                past_condition_latent_src, 
                past_condition_latent_tgt,
            )

            src_latent_list.append(src_latent.clone().to(self.dtype))
            tgt_latent_list.append(tgt_latent.clone().to(self.dtype))

            dir_name_rec = "result_frames_rec" 
            reconstructed_frames = self.decode_latent(torch.cat(src_latent_list, dim=2).clone())
            Path(self.config["output_path"], dir_name_rec).mkdir(parents=True, exist_ok=True)
            reconstructed_frame = reconstructed_frames[-1].resize((self.ori_resolution[1], self.ori_resolution[0]))
            reconstructed_frame.save(Path(self.config["output_path"], dir_name_rec, f"frame_{unit_index:04d}.jpg"))

            # save image
            dir_name = "result_frames"
            reconstructed_frames = self.decode_latent(torch.cat(tgt_latent_list, dim=2).clone())
            Path(self.config["output_path"], dir_name).mkdir(parents=True, exist_ok=True)
            reconstructed_frame = reconstructed_frames[-1].resize((self.ori_resolution[1], self.ori_resolution[0]))
            reconstructed_frame.save(Path(self.config["output_path"], dir_name, f"frame_{unit_index:04d}.jpg"))

        edited_frames = self.decode_latent(torch.cat(tgt_latent_list, dim=2).clone())
        edited_frames = [
            frame.resize((self.ori_resolution[1], self.ori_resolution[0])) 
            for frame in edited_frames
        ]
        
        dir_name = "result_all_frames"
        Path(self.config["output_path"], dir_name).mkdir(parents=True, exist_ok=True)
        for idx, edited_frame in enumerate(edited_frames):
            edited_frame.save(Path(self.config["output_path"], dir_name, f"frame_{idx:04d}.jpg"))

        video_name = "edit.mp4"
        export_to_video(
            edited_frames, 
            Path(self.config["output_path"], video_name), 
            fps=12
        )

        # src videos
        reconstructed_frames = self.decode_latent(torch.cat(src_latent_list, dim=2).clone())
        reconstructed_frames = [
            frame.resize((self.ori_resolution[1], self.ori_resolution[0])) 
            for frame in reconstructed_frames
        ]
        
        dir_name = "result_all_frames_rec"
        Path(self.config["output_path"], dir_name).mkdir(parents=True, exist_ok=True)
        for idx, reconstructed_frame in enumerate(reconstructed_frames):
            reconstructed_frame.save(Path(self.config["output_path"], dir_name, f"frame_{idx:04d}.jpg"))

        video_name = "rec.mp4"
        export_to_video(
            reconstructed_frames, 
            Path(self.config["output_path"], video_name), 
            fps=12
        )

        # combined videos
        combined_latent_list = [
            torch.cat([src, tgt], dim=-1)
            for src, tgt in zip(src_latent_list, tgt_latent_list)
        ]
        combined_frames = self.decode_latent(torch.cat(combined_latent_list, dim=2).clone())
        combined_frames = [
            frame.resize((2*self.ori_resolution[1], self.ori_resolution[0])) 
            for frame in combined_frames
        ]
        video_name = "rec_edit.mp4"
        export_to_video(
            combined_frames, 
            Path(self.config["output_path"], video_name), 
            fps=12
        )

        return tgt_latent_list
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=41)
    parser.add_argument("--data_dir", type=str, default='data/images/')
    parser.add_argument("--config_path", type=str, default="models/pyramid-edit/config.yaml")
    parser.add_argument("--model_name", type=str, default="pyramid_flux", help="pyramid_flux or pyramid_mmdit")  
    parser.add_argument("--model_path", type=str, default="models/pyramid-edit/hf/pyramid-flow-miniflux")
    parser.add_argument("--resolution", type=str, default="384p")
    parser.add_argument("--dataset_json", type=str, default=None, help="json file in FiVE-Bench: data/edit_prompt/edit5_FiVE.json")
    parser.add_argument("--guidance_start_timestep_first", type=int, default=850)
    parser.add_argument("--guidance_stop_timestep_first", type=int, default=100)
    parser.add_argument("--guidance_start_timestep", type=int, default=750)
    parser.add_argument("--guidance_stop_timestep", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--video_guidance_scale", type=float, default=None)
    parser.add_argument("--output_path", type=str, default="outputs/pyramid_edit_results/", help="FiVE dataset json")
    parser.add_argument("--eval_memory_time", action="store_true", help="Enable evaluation of memory time.")
    parser.add_argument("--skip_processed", action="store_true", help="Skip processed videos.")
    # debug
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--source_prompt", type=str, default=None)
    parser.add_argument("--target_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    opt = parser.parse_args()
    
    config = OmegaConf.load(opt.config_path)
    config["max_frames"] = opt.max_frames
    config["data_dir"] = opt.data_dir
    config["model_name"] = opt.model_name
    config["model_path"] = opt.model_path
    config["resolution"] = opt.resolution
    config["dataset_json"] = opt.dataset_json

    if opt.guidance_start_timestep_first > 0:
        config["guidance_start_timestep_first"] = opt.guidance_start_timestep_first
    if opt.guidance_stop_timestep_first > 0:
        config["guidance_stop_timestep_first"] = opt.guidance_stop_timestep_first
    if opt.guidance_start_timestep > 0:
        config["guidance_start_timestep"] = opt.guidance_start_timestep
    if opt.guidance_stop_timestep > 0:
        config["guidance_stop_timestep"] = opt.guidance_stop_timestep

    if opt.guidance_scale:
        config["guidance_scale"] = opt.guidance_scale
    if opt.video_guidance_scale:
        config["video_guidance_scale"] = opt.video_guidance_scale
    if opt.output_path:
        config["output_path"] = opt.output_path.rstrip('/')

    if opt.video_name is not None:
        config["data_path"] = os.path.join(config["data_dir"], opt.video_name)
        if opt.source_prompt:
            config["source_prompt"] = "Photorealistic, high-definition image of " + opt.source_prompt
        if opt.target_prompt:
            config["target_prompt"] = "Photorealistic, high-definition image of " + opt.target_prompt
        if opt.negative_prompt:
            config["negative_prompt"] = opt.negative_prompt
        
        output_path = os.path.join(config["output_path"], opt.video_name, opt.target_prompt[:20].replace(' ', '_'))
        config["output_path"] = f"{output_path}_start_{opt.guidance_start_timestep_first}_{opt.guidance_start_timestep}_stop_{opt.guidance_stop_timestep_first}_{opt.guidance_stop_timestep}_guidance_scale_{opt.guidance_scale}_{opt.video_guidance_scale}"
        Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")

        guidance = Guidance(config)
        tgt_latent_list = guidance.run()

    else:
        with open(opt.dataset_json, 'r') as json_file:
            data = json.load(json_file)

        import psutil, time
        if opt.eval_memory_time:
            data = data[:1]  # GPU/Speed
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 ** 2)  
        start_time = time.time()

        num_videos = len(data)
        output_root = config["output_path"]
        for vid, entry in enumerate(data):
            print(f"Processing {vid}/{num_videos} video: {entry['video_name']} ...")

            config["data_path"] = os.path.join(config["data_dir"], entry['video_name'])
            config["source_prompt"] = entry['source_prompt']
            config["target_prompt"] = entry['target_prompt']
            config["negative_prompt"] = entry['negative_prompt']

            video_name = entry['video_name']
            config["output_path"] = os.path.join(output_root, video_name, entry["save_dir"])
            if opt.skip_processed and os.path.exists(os.path.join(config["output_path"], "edit.mp4")):
                print(f"Video has been processed! Skip {video_name}")
                continue

            Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")

            guidance = Guidance(config)
            tgt_latent_list = guidance.run()

        # save GPU Memory / Speed
        running_time = time.time() - start_time
        max_cpu_memory = process.memory_info().rss / (1024 ** 2)  # to MB

        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 2)  # to MB
        else:
            peak_gpu_memory = 0.0  

        with open(f"{output_root}/memory_stats.txt", "a") as f:
            f.write(f"7-Pyramid-Edit: Max CPU Memory Usage: {max_cpu_memory:.2f} MB\n")
            f.write(f"7-Pyramid-Edit: Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB\n")
            f.write(f"7-Pyramid-Edit: Running Time: {running_time:.2f} seconds\n\n")

        print(f"Max CPU Memory Usage: {max_cpu_memory:.2f} MB")
        print(f"Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB")
        print(f"Running Time: {running_time:.2f} seconds")