#!/bin/bash
set -x
source consts.sh

export CUDA_VISIBLE_DEVICES=0

MOE_LAYERS_LIST=("10")
TASK_NAME=cola
MODEL_NAME=bert-large-cased
NUM_EXPERTS_LIST=(8)
REPEAT=16
SEEDS="0 1 2"
LR="2e-5"
GATE=mintau

for MOE_LAYERS in ${MOE_LAYERS_LIST[@]}; do
    MOE_LAYERS_NAME=$MOE_LAYERS
    for NUM_EXPERTS in ${NUM_EXPERTS_LIST[@]}; do
        echo "GATE: $GATE, MOE_LAYERS: $MOE_LAYERS, MOE_LAYERS_NAME: $MOE_LAYERS_NAME"

        TIME=$(date "+%Y%m%d-%H%M%S")
        output_dir=logs/${TASK_NAME}/${MODEL_NAME}/${GATE}/moe_${NUM_EXPERTS}_experts_adaptive_topk_layers${MOE_LAYERS_NAME}_repeat${REPEAT}/${TIME}
        mkdir -p $output_dir

        nohup python Language/search_glue_no_trainer_mass.py \
            --model_name_or_path $MODEL_NAME \
            --to_MoE \
            --enable_mass \
            --gate_type $GATE \
            --task_name $TASK_NAME \
            --learning_rates $LR \
            --num_experts $NUM_EXPERTS \
            --moe_layers $MOE_LAYERS \
            --seed $SEEDS \
            --expert_repeat $REPEAT \
            --expansion_start_ratio 0.0 \
            --expansion_end_ratio 0.1 \
            --max_expert_num 16 \
            --adaptive_experts \
            --rm_threshold 0.50 \
            --mass_p_threshold 0.01 \
            --mass_similarity_threshold 0.001 \
            --mass_expansion_patience 3 \
            --mass_redundancy_weight 0.01 \
            --random_cluster \
            --save_model > $output_dir/train_nohup.out 2>&1 &
        # wait
    done
done