#!/bin/bash
set -x
source consts.sh
export CUDA_VISIBLE_DEVICES=0

DEVICE=0
SEEDS=(0)
ENVS=(0 1 2 3)
OUTPUT_DIR=outputs

mkdir -p $OUTPUT_DIR/logs

for SEED in "${SEEDS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        nohup python3 -m domainbed.scripts.train_mass \
            --data_dir=/domainbed/data \
            --dataset PACS \
            --seed $SEED \
            --test_envs $ENV \
            --rm_threshold 0.30 \
            --algorithm GMOE \
            --hparams '{"vanilla_ViT":false, "vit_type":"small", "router": "mintau", "adaptive_experts":true, "max_expert_num": 8}' \
            --output_dir $OUTPUT_DIR \
            --device $DEVICE \
            --enable_mass \
            --mass_p_threshold 0.01 \
            --mass_similarity_threshold 0.002 \
            --mass_expansion_patience 3 \
            --mass_redundancy_weight 0.01 > $OUTPUT_DIR/logs/pacs_seed${SEED}_env${ENV}.log 2>&1 &
    done
done
