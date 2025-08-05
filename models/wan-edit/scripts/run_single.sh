CUDA_VISIBLE_DEVICES=7 python models/wan-edit/edit.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 41 \
    --ckpt_dir models/wan-edit/hf/Wan2.1-T2V-1.3B/ \
    --data_dir data/examples \
    --video_dir data/examples \
    --video_name blackswan \
    --save_file outputs/blackswan.mp4 \
    --prompt "A black swan with a red beak swimming in a river near a wall and bushes. Amazing quality, masterpiece." \
    --tgt_prompt "A pink flamingo swimming in a river near a wall and bushes. Amazing quality, masterpiece."

CUDA_VISIBLE_DEVICES=7 python models/wan-edit/edit.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 41 \
    --ckpt_dir models/wan-edit/hf/Wan2.1-T2V-1.3B/ \
    --data_dir data/examples \
    --video_dir data/examples \
    --video_name bear \
    --save_file outputs/bear_pink.mp4 \
    --prompt "A large brown bear is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the bear's deliberate movements." \
    --tgt_prompt "A purple bear is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the bear's deliberate movements."