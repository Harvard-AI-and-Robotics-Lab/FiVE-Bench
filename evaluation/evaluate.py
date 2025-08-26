import json
import argparse
import math
import os
import numpy as np
import glob
import csv
import cv2
import torch
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf


from torchvision.io import read_video
from decord import VideoReader, cpu
import imageio

from metrics_calculator import MetricsCalculator, average_niqe_from_txt


def mask_decode(encoded_mask, image_shape=[512,512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array



def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt, 
                     src_image_path, tgt_image_path, src_save_file_niqe, tgt_save_file_niqe):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="ssim_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    if metric == "niqe_source_image":
        return metrics_calculator.calculate_NIQE(src_save_file_niqe, img_pred_path=src_image_path, img_gt_path=None)
    if metric == "niqe_target_image":
        return metrics_calculator.calculate_NIQE(tgt_save_file_niqe, img_pred_path=None, img_gt_path=tgt_image_path)
    
def calculate_metric_video_level(metrics_calculator, metric, src_video_path, tgt_video_path, 
                                 multiple_choice_question=None, source_yes_no_question=None, target_yes_no_question=None,
                                 tgt_prompt=None, tgt_images=None, tgt_word=None, tgt_video_mask=None,
                                 ):
    if metric in {"motion_fidelity_score", "motion_fidelity_score_edit_part"}:
        return metrics_calculator.calculate_motion_fidelity_score(
            src_video_path, tgt_video_path, 
            video_masks=tgt_video_mask if metric == "motion_fidelity_score_edit_part" else None
        )
    elif metric == "five_acc":
        return metrics_calculator.calculate_five_acc(
            source_yes_no_question, target_yes_no_question, multiple_choice_question, tgt_video_path
        )
    else:
        raise ValueError(f"Metric {metric} not supported")


def list_images(directory):
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    # Create a list to store image paths
    image_files = []
    
    # Loop through each extension and find matching files
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(image_files)

def mp4_to_frames_ffmpeg(video_path):
    output_dir = video_path.replace(".mp4", "")
    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg to extract frames
    output_pattern = os.path.join(output_dir, "%05d.jpg")  # Frame naming pattern
    command = [
        "ffmpeg",
        "-i", video_path,  # Input video file
        output_pattern  # Output frame pattern
    ]

    subprocess.run(command, check=True)
    return output_dir

def calculate_mean(evaluation_result):
    if evaluation_result is None:
        return "nan"
    
    # Filter out 'nan' values
    non_nan_values = [x for x in evaluation_result if x != "nan" and not math.isnan(x)]
    
    # If all values are 'nan', return 'nan'
    if not non_nan_values:
        return "nan"
    
    # Calculate the mean of non-'nan' values
    return sum(non_nan_values) / len(non_nan_values)

    
def main(args, config, all_tgt_video_folders):
    annotation_mapping_files = args.annotation_mapping_files
    metrics = args.metrics
    src_image_folder = args.src_image_folder
    tgt_methods = args.tgt_methods
    edit_category_list = args.edit_category_list
    evaluate_whole_table = args.evaluate_whole_table
    frame_stride = args.frame_stride
    if args.evaluate_source_video:
        tgt_video_folders = {
            "source_videos": (os.path.join(src_image_folder, "images"), "")
        }
        args.result_path = args.result_path.replace(".csv", "_source_videos.csv")
    else:
        tgt_video_folders = {}
        if evaluate_whole_table:
            for key in all_tgt_video_folders:
                if key[0] in tgt_methods:
                    tgt_video_folders[key] = all_tgt_video_folders[key]
        else:
            for key in tgt_methods:
                tgt_video_folders[key] = all_tgt_video_folders[key]
    
    result_path = args.result_path.replace(".csv", f"_frame_stride{frame_stride}.csv")
    result_path_name = result_path.split('/')[-1]
    result_dir = '/'.join(result_path.split('/')[:-1])
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    metrics_calculator = MetricsCalculator(args.device, config=config)
    
    result_avg_files = []
    for annotation_mapping_file in tqdm(annotation_mapping_files, desc="Evaluating annotation mapping files", total=len(annotation_mapping_files)):
        print(f"evaluating {annotation_mapping_file} ...")

        annotation_mapping_file_name = annotation_mapping_file.split("/")[-1].replace(".json", "")
        result_path = os.path.join(
            result_dir, 
            "_".join([annotation_mapping_file_name, result_path_name])
        )

        with open(result_path,'w',newline="") as f:
            csv_write = csv.writer(f)
            
            csv_head = []
            for tgt_video_folder_key, _ in tgt_video_folders.items():
                for metric in metrics:
                    if metric in {"five_acc"}:
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_yes_no")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_multi_choice")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_union")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_inter")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}")
                    else:
                        csv_head.append(f"{tgt_video_folder_key}|{metric}")
            
            data_row = ["file_id"] + csv_head
            csv_write.writerow(data_row)

        with open(annotation_mapping_file, "r") as f:
            annotation_file = json.load(f)

        for key, item in tqdm(enumerate(annotation_file), desc="Evaluating videos", total=len(annotation_file)):
            if str(item["editing_type_id"]) not in edit_category_list:
                continue

            video_name = item["video_name"]
            save_dir = str(item["editing_type_id"]) + "_" + item["target_prompt"][:len(item["save_dir"])-2]  # item["save_dir"]
            source_prompt = item["source_prompt"].replace("[", "").replace("]", "")
            target_prompt = item["target_prompt"].replace("[", "").replace("]", "")
            # FiVE_acc
            # "multiple_choice_question": "Is the cyclist wearing a helmet? \na) Yes \nb) No",
            # "source_yes_no_question": "Is the cyclist wearing a helmet in the image?",
            # "target_yes_no_question": "Is the cyclist not wearing a helmet in the image?"
            if "multiple_choice_question" in item:
                multiple_choice_question = item["multiple_choice_question"]
                source_yes_no_question = item["source_yes_no_question"]
                target_yes_no_question = item["target_yes_no_question"]
            else:
                multiple_choice_question = None
                source_yes_no_question = None
                target_yes_no_question = None

            src_video_path = os.path.join(src_image_folder, "images", video_name)
            src_image_names = list_images(src_video_path)[::frame_stride]
            if args.evaluate_source_video:
                src_image_names = src_image_names[:40//frame_stride]

            src_images = [
                Image.open(src_image_name)
                for src_image_name in src_image_names
            ]

            mask_path = os.path.join(src_image_folder, "bmasks", video_name)
            if not os.path.exists(mask_path):
                print(f"{video_name}'s mask cannot be found!! Skip ...")
                continue

            masks = []
            for src_image_name in src_image_names:
                mask = Image.open(os.path.join(mask_path, src_image_name.split('/')[-1]))

                # Convert the mask to a numpy array and ensure it's binary (0 and 1)
                # mask = mask_decode(item["mask"])
                mask = np.array(mask)  # Convert to numpy array
                mask = (mask > 0)
                mask = mask[:,:,np.newaxis].repeat([3],axis=2)
                masks.append(mask)
        
            evaluation_result = [key]
            
            for m_i, (tgt_video_folder_key, (tgt_video_folder, terminal_folder)) in enumerate(tgt_video_folders.items()):
                src_save_file_niqe = "_".join([
                    result_path.replace(".csv", ""), "niqe_src.txt"
                ])
                tgt_save_file_niqe = "_".join([
                    result_path.replace(".csv", ""), "niqe_"+tgt_video_folder_key+"_tgt.txt"
                ])

                if not args.evaluate_source_video:
                    if tgt_video_folder_key != "6_VideoGrain":
                        tgt_video_name = os.path.join(video_name, save_dir, terminal_folder)  # terminal_folder = "image_ode" in TokenFlow
                    else:
                        prefix = annotation_mapping_file.split('/')[-1][:5]
                        assert prefix.startswith("edit")
                        tgt_video_name = os.path.join(prefix, video_name)
                    tgt_video_path = os.path.join(tgt_video_folder, tgt_video_name)
                else:
                    tgt_video_path = src_video_path
                print(f"\n\nevaluating method: {tgt_video_folder_key}")
                
                if tgt_video_path.endswith("/"):
                    tgt_video_path = tgt_video_path[:-1]
                tgt_video_path_mp4 = tgt_video_path + '.mp4'
                if os.path.exists(tgt_video_path_mp4):   
                    # NOTE: must use ffmpeg!!
                    tgt_video_path = mp4_to_frames_ffmpeg(tgt_video_path_mp4)

                tgt_image_names = list_images(tgt_video_path)
                tgt_image_names = tgt_image_names[::frame_stride]
                tgt_images = []
                for f_i, tgt_image_name in enumerate(tgt_image_names):
                    if tgt_image_name.endswith(".jpg") or tgt_image_name.endswith(".png"):
                        tgt_image = Image.open(tgt_image_name).resize(src_images[0].size)  
                        tgt_images.append(tgt_image)

                        tgt_image_name = os.path.join(
                            "/".join(tgt_image_name.split('/')[:-1])+"_resize", 
                            os.path.basename(tgt_image_name)
                        )
                        tgt_image_names[f_i] = tgt_image_name
                        Path("/".join(tgt_image_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                        tgt_image.save(tgt_image_name)
                
                for m_j, metric in enumerate(metrics):
                    if metric in {"niqe_source_image"} and m_i > 0:
                        continue

                    print(f"\nevaluating metric: {metric}")
                    if len(tgt_images) == 0:
                        print(f"No images are founded {tgt_video_path}! Skip ...")
                        if metric in {"five_acc"}:
                            evaluation_result += ["nan"] * 5
                        else:
                            evaluation_result.append("nan")
                        continue
                    
                    assert len(os.listdir(src_video_path)) > 0 and \
                        len(tgt_images) > 0, f"No images are founded!"

                    try:
                        if metric in {"motion_fidelity_score", "motion_fidelity_score_edit_part", "five_acc"}:
                            if args.evaluate_source_video:
                                eval_result_ = (
                                    calculate_metric_video_level(
                                        metrics_calculator, metric,
                                        src_video_path, src_video_path,
                                        multiple_choice_question=multiple_choice_question,
                                        source_yes_no_question=source_yes_no_question,
                                        target_yes_no_question=target_yes_no_question,
                                        tgt_video_mask=masks
                                    )
                                )
                            else:
                                eval_result_ = (
                                    calculate_metric_video_level(
                                        metrics_calculator, metric,
                                        src_video_path, tgt_video_path,
                                        multiple_choice_question=multiple_choice_question,
                                        source_yes_no_question=source_yes_no_question,
                                        target_yes_no_question=target_yes_no_question,
                                        tgt_video_mask=masks
                                    )
                                )
                            # Five_acc ouputs YN-acc and MC-acc 
                            if metric in {"five_acc"}:
                                if "nan" in eval_result_:
                                    evaluation_result += ["nan"] * 5
                                else:
                                    eval_result_ =  list(eval_result_)
                                    evaluation_result_five = []
                                    for eval_result_s in list(eval_result_):
                                        evaluation_result_five.append(eval_result_s)
                                    evaluation_result_five.append(int(sum(eval_result_) > 0))
                                    evaluation_result_five.append(int(sum(eval_result_) >= len(eval_result_)))
                                    evaluation_result_five.append(calculate_mean(evaluation_result_five))
                                    evaluation_result += evaluation_result_five
                            else:
                                evaluation_result.append(eval_result_)

                        else:
                            
                            if metric in {"niqe_source_image", "niqe_target_image"}:
                                if os.path.exists(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe):
                                    os.remove(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe)

                            evaluation_result_each_frame = []
                            for src_image, tgt_image, mask, src_image_path, tgt_image_path, in zip(src_images[:len(tgt_images)], tgt_images, masks, src_image_names[:len(tgt_images)], tgt_image_names):
                                assert src_image.size[0] == tgt_image.size[0] and src_image.size[1] == tgt_image.size[1], \
                                    f"{tgt_video_folder_key}: {src_image.size} != {tgt_image.size})"
                                
                                if args.evaluate_source_video:
                                    evaluation_result_each_frame.append(
                                        calculate_metric(
                                            metrics_calculator, metric, 
                                            src_image, src_image, 
                                            mask, mask, 
                                            source_prompt, target_prompt,
                                            src_image_path, src_image_path, 
                                            src_save_file_niqe, src_save_file_niqe,
                                        )
                                    )
                                else:
                                    evaluation_result_each_frame.append(
                                        calculate_metric(
                                            metrics_calculator, metric, 
                                            src_image, tgt_image, 
                                            mask, mask, 
                                            source_prompt, target_prompt,
                                            src_image_path, tgt_image_path, 
                                            src_save_file_niqe, tgt_save_file_niqe,
                                        )
                                    )
                            
                            if metric in {"niqe_source_image", "niqe_target_image"}:
                                evaluation_result.append(
                                    average_niqe_from_txt(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe)
                                )
                            else:
                                evaluation_result.append(
                                    calculate_mean(evaluation_result_each_frame)
                                )

                    except Exception as e:
                        print(f"Error: {metric}: {e}")
                        continue
                      
            with open(result_path, 'a+', newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)
 
        # calculate the average of each metric (each column)
        with open(result_path, 'r') as f:
            reader = list(csv.reader(f))
            header, rows = reader[0], reader[1:]

        avg_row = []
        # Process each column by index to handle rows with different lengths
        for col_idx, name in enumerate(header):
            print("processing", name)
            # Extract column values, handling missing values
            col_values = []
            for row in rows:
                if col_idx < len(row):
                    col_values.append(row[col_idx])
                else:
                    col_values.append("")  # Use empty string for missing values
            
            try:
                # Filter out empty strings and convert to float
                values = [float(x) for x in col_values if x != "" and x != "nan"]
                if values:  # Only calculate average if there are valid values
                    avg = sum(values) / len(values)
                    if 'structure_distance' in name:
                        avg *= 1000
                    elif 'lpips_' in name:
                        avg *= 1000
                    elif 'mse_' in name:
                        avg *= 10000
                    elif 'ssim_' in name:
                        avg *= 100
                    elif 'motion_fidelity_score' in name:
                        avg *= 100
                    elif name.startswith('five_acc'):
                        avg *= 100
                    avg_row.append(f"{avg:.4f}")
                else:
                    avg_row.append("N/A")
            except ValueError:
                avg_row.append("N/A")

        result_avg_files.append(result_path.replace('.csv', '_avg.csv'))
        with open(result_avg_files[-1], 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerow(avg_row)
    
    # average the results in result_avg_files
    if result_avg_files:
        all_avg_rows = []
        
        # Read all average files
        for result_avg_file in result_avg_files:
            with open(result_avg_file, 'r') as f:
                reader = list(csv.reader(f))
                header, rows = reader[0], reader[1:]
                if rows:  # Make sure there's data
                    all_avg_rows.append(rows[0])  # Get the average row
        
        # Calculate final averages across all files
        final_avg_row = []
        for col_idx, name in enumerate(header):
            print("final averaging", name)
            
            # Extract values from all average files for this column
            col_values = []
            for avg_row in all_avg_rows:
                if col_idx < len(avg_row) and avg_row[col_idx] != "N/A":
                    try:
                        col_values.append(float(avg_row[col_idx]))
                    except ValueError:
                        pass  # Skip non-numeric values
            
            # Calculate final average
            if col_values:
                final_avg = sum(col_values) / len(col_values)
                final_avg_row.append(f"{final_avg:.4f}")
            else:
                final_avg_row.append("N/A")
        
        # Write final averaged results
        with open(f"{os.path.dirname(result_avg_files[0])}/final_averaged_results.csv", 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerow(final_avg_row)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument('--annotation_mapping_files', nargs = '+', type=str, default=[
                                                        "data/edit_prompt/edit1_FiVE.json",
                                                        "data/edit_prompt/edit2_FiVE.json",
                                                        "data/edit_prompt/edit3_FiVE.json",
                                                        "data/edit_prompt/edit4_FiVE.json",
                                                        "data/edit_prompt/edit5_FiVE.json",
                                                        "data/edit_prompt/edit6_FiVE.json",
                                                        ])
    parser.add_argument('--metrics', nargs = '+', type=str, default=[
                                                         "structure_distance",
                                                         "psnr_unedit_part",
                                                         "lpips_unedit_part",
                                                         "mse_unedit_part",
                                                         "ssim_unedit_part",
                                                         "clip_similarity_source_image",
                                                         "clip_similarity_target_image",
                                                         "clip_similarity_target_image_edit_part",
                                                        #  "niqe_source_image",
                                                         "niqe_target_image",
                                                         "motion_fidelity_score",
                                                         "motion_fidelity_score_edit_part",
                                                         "five_acc",    
                                                        ])
    parser.add_argument('--src_image_folder', type=str, default="data/")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=[
                                                                    # "1_TokenFlow",
                                                                    # "2_DMT",
                                                                    # "4_VidToMe",
                                                                    # "5_AnyV2V",
                                                                    # "6_VideoGrain",
                                                                    # "7_Pyramid_Edit",
                                                                    "8_Wan_Edit",
                                                                  ])
    parser.add_argument('--result_path', type=str, default="outputs/evaluation_result.csv")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--edit_category_list',  nargs = '+', type=str, default=[
                                                                                "1",
                                                                                "2",
                                                                                "3",
                                                                                "4",
                                                                                "5",
                                                                                "6",
                                                                                ]) # the editing category that needed to run
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images
    parser.add_argument('--evaluate_source_video', action= "store_true")
    parser.add_argument('--config_path', type=str, default="config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key in config and value is not None:
            config[key] = value
    
    # NOTE: Modify the target video folders here!!!!! 
    all_tgt_video_folders = {
        # "1_TokenFlow": (f"{config.root_tgt_video_folder}/TokenFlow/", "img_ode"),
        # "2_DMT": (f"{config.root_tgt_video_folder}/diffusion-motion-transfer/", "result_frames"),
        # "4_VidToMe": (f"{config.root_tgt_video_folder}/VidToMe/", "frames"),
        # "5_AnyV2V": (f"{config.root_tgt_video_folder}/AnyV2V/Results/Prompt-Based-Editing_frames32/i2vgen-xl", "ddim_init_latents_t_idx_0_nsteps_50_cfg_9.0_pnpf0.2_pnps0.2_pnpt0.5"),
        # "6_VideoGrain": (f"{config.root_tgt_video_folder}/video_grain/", ""),
        "7_Pyramid_Edit": (f"{config.root_tgt_video_folder}/Pyramid-edit/", "result_all_frames"),
        "8_Wan_Edit": (f"{config.root_tgt_video_folder}/Wan-Edit/", ""),
    }

    main(args, config, all_tgt_video_folders)