# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import cv2
import json
import numpy as np
warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50


    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="The file to save the video needed to be edited.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs",
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--tgt_prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--tgt_guide_scale",
        type=float,
        default=10.0,
        help="Target guide scale for Wan-Edit.")
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        default=16,
        help="Skip timesteps for Wan-Edit.")
    
    # FiVE
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data")
    parser.add_argument(
        "--video_name",
        type=str,
        default=None)
    parser.add_argument(
        "--FiVE_dataset_json",
        type=str,
        default=None,
        help="dataset json: data_FiVE/edit_prompt/edit1_FiVE.json, including src, tgt promts")
    parser.add_argument(
        "--eval_gpu_time",
        type=bool,
        default=False,
        help="if enable, it will be used to test GPU memory and running time.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def load_frames(video_path=None, num_frames=41, target_size=(832, 480)):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    # Check if video is successfully opened
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    frames = []
    # Read first num_frames frames
    for i in range(num_frames):
        ret, frame = cap.read()
        # If video ends, exit loop early
        if not ret:
            break
        # Resize frame
        resized_frame = cv2.resize(frame, target_size)
        # Convert frame from BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to tensor and normalize [-1, 1]
        tensor_frame = torch.tensor(resized_frame).permute(2, 0, 1).float() / 255.0
        tensor_frame = 2 * tensor_frame - 1
        # Add to frame list
        frames.append(tensor_frame)
    # Release video object
    cap.release()
    # Stack frame list into tensor
    if frames:
        frames_tensor = torch.stack(frames).permute(1,0,2,3)
    else:
        raise ValueError("Video does not have enough frames")
    return frames_tensor.unsqueeze(0)

def load_frames_path(video_path=None, num_frames=41, target_size=(832, 480)):
    frame_files = sorted(os.listdir(video_path))  # Get and sort frame filenames
    frame_files = [f for f in frame_files if f.endswith('.jpg') or f.endswith('.png')]  # Ensure only .jpg and .png files are selected

    frames = []
    for i in range(min(num_frames, len(frame_files))):  # Read specified number of frames in order
        frame_path = os.path.join(video_path, frame_files[i])
        
        # Use OpenCV to read image
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Cannot read image: {frame_path}")
            continue
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # resize image 
        frame = cv2.resize(frame, target_size)
        
        # Convert to float and normalize to [-1, 1]
        frame = 2 * frame.astype(np.float32) / 255.0 - 1
        
        # Adjust dimension order to [C, H, W]
        frame = np.transpose(frame, (2, 0, 1))
        
        frames.append(frame)

    # Convert frame list to tensor
    frames_tensor = torch.tensor(np.array(frames)).float()
    frames_tensor = frames_tensor.permute(1,0,2,3).unsqueeze(0)
    return frames_tensor


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.video_path.endswith('.mp4'):
        video = load_frames(args.video_path)
    elif os.path.isdir(args.video_path):
        video = load_frames_path(args.video_path)
    else:
        raise ValueError(f"Invalid video path: {args.video_path}")

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")

        video = wan_t2v.edit(
            video,
            args.prompt,
            args.tgt_prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=min(args.frame_num, video.shape[2]),
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            tgt_guide_scale=args.tgt_guide_scale,
            skip_timesteps=args.skip_timesteps,
            seed=args.base_seed,
            offload_model=args.offload_model)
        
    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()

    if args.FiVE_dataset_json is None:
        args.video_path = os.path.join(
            args.video_dir,
            args.video_name
        )
        generate(args)

    else:
        with open(args.FiVE_dataset_json, 'r') as file:
            data = json.load(file)

        # GPU/Speed
        import psutil, time
        if args.eval_gpu_time:
            data = data[:1]
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 ** 2)  
        start_time = time.time()

        filed_videos = []
        num_videos = len(data)
        for vid, entry in enumerate(data):
            video_name = entry["video_name"]
            print(f"Processing {vid}/{num_videos} video: {video_name} ...")

            args.prompt = entry["source_prompt"]
            args.tgt_prompt = entry["target_prompt"]
            args.video_path = os.path.join(
                args.data_dir, 
                entry["video_name"]+'.mp4'
            )
            type_idx = args.FiVE_dataset_json.split('/')[-1].split('_')[0].replace("edit", "")
            args.save_file = os.path.join(
                args.save_dir, 
                entry["video_name"],
                type_idx + '_' + entry["target_prompt"][:20]+'.mp4'
            )

            if os.path.exists(args.save_file):
                print(f"The video has been edited! Skip {args.save_file}")
                continue

            try: 
                generate(args)
            except Exception as e:
                print(f"Error: {e}")  
                filed_videos.append(vid)
                continue

        # save GPU Memory / Speed
        running_time = time.time() - start_time
        max_cpu_memory = process.memory_info().rss / (1024 ** 2)  # to MB

        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 2)  # to MB
        else:
            peak_gpu_memory = 0.0  

        with open(f"{args.save_dir}/memory_stats.txt", "a") as f:
            f.write(f"8-Wan-Edit: Max CPU Memory Usage: {max_cpu_memory:.2f} MB\n")
            f.write(f"8-Wan-Edit: Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB\n")
            f.write(f"8-Wan-Edit: Running Time: {running_time:.2f} seconds\n\n")

        print(f"Max CPU Memory Usage: {max_cpu_memory:.2f} MB")
        print(f"Peak GPU Memory Usage: {peak_gpu_memory:.2f} MB")
        print(f"Running Time: {running_time:.2f} seconds")

        print(f"failed videos: {filed_videos}")