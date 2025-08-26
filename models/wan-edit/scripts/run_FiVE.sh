export CUDA_VISIBLE_DEVICES=0

skip_timesteps=15

for i in {1..6}; do
    python models/wan-edit/edit.py \
        --task t2v-1.3B \
        --size 832*480 \
        --frame_num 41 \
        --skip_timesteps ${skip_timesteps} \
        --ckpt_dir models/wan-edit/hf/Wan2.1-T2V-1.3B/ \
        --data_dir data/videos \
        --save_dir outputs/wan_edit_results \
        --FiVE_dataset_json data/edit_prompt/edit${i}_FiVE.json
done