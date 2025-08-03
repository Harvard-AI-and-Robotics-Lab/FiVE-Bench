import os
import torch
import imageio
import subprocess
from torchvision.transforms import Resize
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

try:
    from cotracker.predictor import CoTrackerPredictor
    from cotracker.utils.visualizer import read_video_from_path
except:
    print("No found cotracker, skipped!")

from transformers import AutoProcessor, AutoModel 
from qwen_vl_utils import process_vision_info


def find_images_in_dir(directory):
    assert os.path.isdir(directory), f"{directory}"
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_files = sorted([
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(image_extensions)
    ])
    return image_files

def average_niqe_from_txt(save_file):
    values = []

    with open(save_file, 'r') as file:
        for line in file:
            parts = line.strip().split(",")  # Split by comma
            if len(parts) == 2:  # Ensure there are two parts
                try:
                    values.append(float(parts[1]))  # Extract number and convert to float
                except ValueError:
                    continue  # Skip lines that do not match expected format

    # Compute the average
    average = sum(values) / len(values) if values else 0

    print(f"Total Number of Frames: {len(values)}, Average NIQE Score: {average}")

    return average


class FiVEAcc_Qwen_VL(torch.nn.Module):
    def __init__(self, num_frames=4, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__()

        # num frames are fed into Qwen-VL
        self.num_frames = num_frames

        # different transformer version
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_template(self, q, q_type="yes/no"):
        if q_type == "yes/no":
            input_text = (
                "Answer the following question using only 'YES' or 'NO:\n"
                f"{q}"
            )
        elif q_type == "multi-choice":
            input_text = (
                "Select the correct answer from the given choices, onlyt output the answer:\n"
                f"{q}"
            )
        else:
            raise NotImplementedError

        return input_text
        
    def run_each_iter(self, text, video_path):
        if os.path.isdir(video_path):
            video_path = find_images_in_dir(video_path)
        
        if isinstance(video_path, list):
            stride = len(video_path)//(len(video_path)//self.num_frames)
            video_path = video_path[int(0.5*stride)::stride]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            # "video": [
                            #     "file:///path/to/frame1.jpg",
                            #     "file:///path/to/frame2.jpg",
                            #     "file:///path/to/frame3.jpg",
                            #     "file:///path/to/frame4.jpg",
                            # ],
                            "video": video_path,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        elif video_path.endswith('.mp4'):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            # "video": "file:///path/to/video1.mp4",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            assert video_path.endswith('.jpg') or video_path.endswith('png'), \
            f"unsupported file format {video_path}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                            "image": video_path,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=10, ## Important!!
            padding=True,
            return_tensors="pt",
            # **video_kwargs
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


    def get_score(self, src_q, tgt_q, multi_choice_q, video_path):
        """
        Evaluate answers for source, target, and multiple-choice questions.

        Args:
            src_q (str): Source question.
            tgt_q (str): Target question.
            multi_choice_q (str): Multiple-choice question.
            video_path (str): Path to the video file.

        Returns:
            tuple: A tuple containing:
                - yn_acc (bool): Whether the yes/no answers are correct.
                - mc_acc (bool): Whether the multiple-choice answer is correct.
        """
        assert tgt_q is not None and multi_choice_q is not None
        assert len(tgt_q) > 0 and len(multi_choice_q) > 0
        print(src_q, tgt_q, multi_choice_q)

        try:
            # Process multiple-choice question
            multi_choice_q = self.get_template(multi_choice_q, q_type="multi-choice")
            multi_choice_a = self.run_each_iter(multi_choice_q, video_path)
            mc_acc = multi_choice_a.strip()[:1].lower() == "b"  # Check if the answer is "B"
            print("mc:", multi_choice_a)

            # Process source question
            if src_q is not None and len(src_q) > 0:
                src_q = self.get_template(src_q, q_type="yes/no")
                src_a = self.run_each_iter(src_q, video_path)
                print("src_a:", src_a)
                src_a_cleaned = src_a.strip()[:2].lower()  # Clean and normalize source answer

            # Process target question
            tgt_q = self.get_template(tgt_q, q_type="yes/no")
            tgt_a = self.run_each_iter(tgt_q, video_path)
            print("tgt_a:", tgt_a)

            tgt_a_cleaned = tgt_a.strip()[:3].lower()  # Clean and normalize target answer

            # Evaluate yes/no answers
            if src_q is not None and len(src_q) > 0:
                yn_acc = (src_a_cleaned == "no" and tgt_a_cleaned == "yes")
            else:
                yn_acc = tgt_a_cleaned == "yes"
            print("yn / mc: ", int(yn_acc), int(mc_acc))
            
            return int(yn_acc), int(mc_acc)

        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"An error occurred: {e}")
            return "nan", "nan"  # Return default values in case of an error
        

class MotionFidelityScore(torch.nn.Module):
    def __init__(self, cotracker_model_path):
        super().__init__()

        self.model = CoTrackerPredictor(checkpoint=cotracker_model_path)
        self.model = self.model.cuda()

    def get_similarity_matrix(self, tracklets1, tracklets2):
        displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
        displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

        displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
        displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

        similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
        return similarity_matrix

    def get_score(self, similarity_matrix):
        similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
        # for each row find the most similar element
        max_similarity, _ = similarity_matrix_eye.max(dim=1)
        average_score = max_similarity.mean()
        return {
            "average_score": average_score.item(),
        }

    def read_frames_from_dir(self, dir_path):
        """
        Read frames from a directory of images.

        Parameters:
        - dir_path (str): Path to the directory containing image frames.

        Returns:
        - np.ndarray: A NumPy array of frames, or None if the directory is empty or invalid.
        """
        try:
            # List all image files in the directory (sorted for consistent ordering)
            image_files = sorted(
                [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            )
            if not image_files:
                print(f"No image files found in directory: {dir_path}")
                return None

            # Load all images into a list
            frames = [imageio.imread(img) for img in image_files]
            return np.stack(frames)
        except Exception as e:
            print("Error reading frames from directory:", e)
            return None

    def get_tracklets(self, video_path, mask=None, dw8_after_video_vae=False, cut_frames=None):
        if video_path.endswith('.mp4'):
            video = read_video_from_path(video_path)
        else:
            assert os.path.isdir(video_path), f'{video_path} must be a dir!'
            video = self.read_frames_from_dir(video_path) # t, h, w, 3
        if cut_frames is not None:
            video = video[:cut_frames]

        len_video = len(video)
        if dw8_after_video_vae:
            video = video[::8]  # downsampling ratio of video vae 
        
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
        pred_tracks_small, pred_visibility_small = self.model(video, grid_size=55, segm_mask=mask)
        pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
        return pred_tracks_small, len_video

    def calculate_MFS(self, original_video_path, edit_video_path, video_masks=None, dw8_after_video_vae=False):
        """
            Args:
                video_masks: 0 or 1 mask, 0 for background, 1 for foreground
                dw8_after_video_vae: enable downsample 8x, cause video_vae has 8x temporal downsample 

        """

        if video_masks is not None:  # calculate trajectories only on the foreground of the video
            if isinstance(video_masks, list):
                minx_list, maxx_list, miny_list, maxy_list = [], [], [], []
                for segm_mask in video_masks:
                    if segm_mask.ndim == 3 and segm_mask.shape[-1] == 3:
                        segm_mask = segm_mask[..., 0]
                    assert segm_mask.ndim == 2
                    if isinstance(segm_mask, np.ndarray):
                        segm_mask = torch.from_numpy(segm_mask).float() 
                    minx = segm_mask.nonzero(as_tuple=False)[:, 0].min()
                    maxx = segm_mask.nonzero(as_tuple=False)[:, 0].max()
                    miny = segm_mask.nonzero(as_tuple=False)[:, 1].min()
                    maxy = segm_mask.nonzero(as_tuple=False)[:, 1].max()
                    minx_list.append(minx)
                    maxx_list.append(maxx)
                    miny_list.append(miny)
                    maxy_list.append(maxy)

                # get bounding box mask from segmentation mask - rectangular mask that covers the segmentation mask
                minx, maxx = min(minx_list), max(maxx_list)
                miny, maxy = min(miny_list), max(maxy_list)
                box_mask = torch.zeros_like(segm_mask)
                box_mask[minx:maxx, miny:maxy] = 1
                box_mask = box_mask[None, None]
            else:
                raise ValueError("video_masks must be a list")
                
        else:
            box_mask = None
        
        edit_tracklets, len_video_edit = self.get_tracklets(edit_video_path, mask=box_mask)
        original_tracklets, len_video_ori = self.get_tracklets(
            original_video_path, mask=box_mask, dw8_after_video_vae=dw8_after_video_vae, cut_frames=len_video_edit
        )
        assert len_video_edit == len_video_ori

        similarity_matrix = self.get_similarity_matrix(edit_tracklets, original_tracklets)
        similarity_scores_dict = self.get_score(similarity_matrix)

        return similarity_scores_dict["average_score"]



class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        self.device=device

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = self.attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map
    
    def attn_cosine_sim(self,x, eps=1e-08):
        x = x[0]  # TEMP: getting rid of redundant dimension, TBF
        norm1 = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
        sim_matrix = (x @ x.permute(0, 2, 1)) / factor
        return sim_matrix
    

class LossG(torch.nn.Module):
    def __init__(self, cfg,device):
        super().__init__()

        self.cfg = cfg
        self.device=device
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_entire_cls=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0

    def forward(self, outputs, inputs):
        self.update_lambda_config(inputs['step'])
        losses = {}
        loss_G = 0

        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
            loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        if self.lambdas['lambda_entire_cls'] > 0:
            losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs['x_entire'], inputs['B_global'])
            loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        if self.lambdas['lambda_global_identity'] > 0:
            losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
            loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss
    

class MetricsCalculator:
    def __init__(self, device, config) -> None:
        self.device=device
        self.config = config
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.structure_distance_metric_calculator = LossG(cfg={
                            'dino_model_name': 'dino_vitb8', # ['dino_vitb8', 'dino_vits8', 'dino_vitb16', 'dino_vits16']
                            'dino_global_patch_size': 224,
                            'lambda_global_cls': 10.0,
                            'lambda_global_ssim': 1.0,
                            'lambda_global_identity': 1.0,
                            'entire_A_every':75,
                            'lambda_entire_cls':10,
                            'lambda_entire_ssim':1.0
                        }, device=device)
        
        try:
            self.motion_fidelity_score_calculator = MotionFidelityScore(
                cotracker_model_path=config.cotracker_model_path
            )
        except Exception as e:
            print("Error: ", e)
            print("Failed to load MotionFidelityScore!")
            exit()

        try:
            self.five_acc_calculator = FiVEAcc_Qwen_VL(
                num_frames=config.five_acc_vlm_num_frames,
                model_id=config.five_acc_vlm_model_id
            )
        except Exception as e:
            print("Error: ", e)
            print("Failed to load FiVEAcc_Qwen_VL")
            exit()
    
    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)
        
        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)
            
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score = self.psnr_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).to(self.device)
            
        score =  self.mse_metric_calculator(img_pred_tensor.contiguous(),img_gt_tensor.contiguous())
        score = score.cpu().item()
        
        return score
    
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
        
    def calculate_structure_distance(self, img_pred, img_gt, mask_pred=None, mask_gt=None, use_gpu = True):
        img_pred = np.array(img_pred).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        
        img_pred = torch.from_numpy(np.transpose(img_pred, axes=(2, 0, 1))).to(self.device)
        img_gt = torch.from_numpy(np.transpose(img_gt, axes=(2, 0, 1))).to(self.device)
        img_pred = torch.unsqueeze(img_pred, 0)
        img_gt = torch.unsqueeze(img_gt, 0)
        
        structure_distance = self.structure_distance_metric_calculator.calculate_global_ssim_loss(img_gt, img_pred)
        
        return structure_distance.data.cpu().numpy()

    def calculate_NIQE(self, save_file, img_pred_path=None, img_gt_path=None, use_gpu=True):
        assert img_pred_path is not None or img_gt_path is not None

        model = "NIQE"
        image_path = img_pred_path if img_pred_path is not None else img_gt_path
        # Construct the command
        IQA_PyTorch_model_path = self.config.IQA_PyTorch_model_path
        command = f'python {IQA_PyTorch_model_path}/inference_iqa.py -m {model} -t "{image_path}" --save_file "{save_file}"'
        print(f"Running command: {command}")
        # Run the command and capture output
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
        
        return "nan"
    
    def calculate_motion_fidelity_score(self, original_video_path, edit_video_path, video_masks=None, dw8_after_video_vae=False):
        return self.motion_fidelity_score_calculator.calculate_MFS(
            original_video_path, edit_video_path, video_masks, dw8_after_video_vae
        ) 

    def calculate_five_acc(self, src_q, tgt_q, multi_choice_q, video_path):
        return self.five_acc_calculator.get_score(src_q, tgt_q, multi_choice_q, video_path)