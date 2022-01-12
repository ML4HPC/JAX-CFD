CUDA_VISIBLE_DEVICES=1 python generate_data.py \
    --high_res 1024 --outer_steps 500 \
    --warmup_time 4.2 --max_velocity 4.5 \
    --demo_steps 1000 --decay \
    --demo_file "re1000_decay" \
    --demo
