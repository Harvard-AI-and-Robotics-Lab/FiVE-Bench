CUDA_VISIBLE_DEVICES=6 python models/pyramid-edit/edit.py \
    --dataset_json data/edit_prompt/edit5_FiVE.json \
    --guidance_start_timestep_first 750 \
    --guidance_stop_timestep_first 100 \
    --guidance_start_timestep 750 \
    --guidance_stop_timestep 100 \
    --guidance_scale 7.0 \
    --video_guidance_scale 5.0