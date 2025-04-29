#!/bin/bash

N=1024                          # 1024
EMBED_Modelnet='adaptive' # ('adaptive' 'gaussian' 'cosine')                # ('gaussian' 'grouped_conv_mlp' 'cosine' 'mlp')   # mlp
EMBED_Scanobject='adaptive'
INITIAL_DIM=3                   # 3
EMBED_DIM=32 # 32 64) # (16 32) # 64)            # 16
ALPHA_BETA='no'         #('yes_ba' 'no' 'yes_ab')     # 'yes_ab'         # no
SIGMA=0.4            # 0.4
ALPHA=1000.0          # 100.0
BETA=100.0             # 1.0

RES_DIM_RATIO=0.25
BIAS=false    #####################                  # Flase
USE_XYZ=true  #####################                  # True
NORM_MODE='anchor-yes_ab'       # ('anchor-yes_ab' 'anchor-no' 'anchor-yes_ba')         # 'anchor-no' ('nearest_to_mean' 'anchor' 'center')
STD_MODE='BN11'            # ('BN1D' 'BN11' '1111' 'B111')   # 'B111'          ######## 2

DIM_RATIO='2-2-2-1'                # '2-2-2-1'

NUM_BLOCKS1='1-1-2-1'                # '1-1-2-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'         # 'mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'           # 'mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'        # '1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K_ModelNet='32-32-32-32'
K_ScanObject='24-24-24-24'
SAMPLING_MODE='fps-fps-fps-fps'       # 'fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'                # '2-2-2-2'

CLASSIFIER_MODE='mlp_very_large'  # 'mlp_very_very_large' 'mlp_very_large' 'mlp_large' 'mlp_medium' 'mlp_small' 'mlp_very_small'

BATCH_SIZE_ModelNet=50
BATCH_SIZE_ScanObject=50

EPOCH_ModelNet=1        #300
EPOCH_ScanObject=200

MIN_LR_ModelNet=0.005
MIN_LR_ScanObject=0.005

OPTIMIZER="sgd"  # '[sgd, adamW]'

LEARNING_RATE_SGD=0.1
WEIGHT_DECAY_SGD=2e-4

LEARNING_RATE_ADAMW=1e-3
WEIGHT_DECAY_ADAMW=1e-2

SCHEDULER='CosineAnnealingLR'   # [CosineAnnealingLR, CosineAnnealing_Warmup]
CRITERION='cal_loss'            # [cal_loss, CrossEntropy_Smoothing]

ROTATION_HEAD='no'              # 'no'

# SEED=(42 52 62 72 82 92 12 22 32 100 200 400 600 1000 2000 5000)
WORKERS_ModelNet=6
WORKERS_ScanObject=6
EMA='no'


for i in {1..100}; do

        python cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" \
        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
        --batch_size "$BATCH_SIZE_ModelNet" --epoch "$EPOCH_ModelNet" \
        --optimizer "$OPTIMIZER" \
        --learning_rate_sgd "$LEARNING_RATE_SGD" --weight_decay_sgd "$WEIGHT_DECAY_SGD" \
        --learning_rate_adamw "$LEARNING_RATE_ADAMW" --weight_decay_adamw "$WEIGHT_DECAY_ADAMW" \
        --scheduler "$SCHEDULER" --criterion "$CRITERION" \
        --min_lr "$MIN_LR_ModelNet" \
        --workers "$WORKERS_ModelNet" \
        --alpha_beta "$ALPHA_BETA" --ema "$EMA" --rotation_head "yes"
        echo "====================================================================="


        python cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" \
        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
        --batch_size "$BATCH_SIZE_ModelNet" --epoch "$EPOCH_ModelNet" \
        --optimizer "adamw" \
        --learning_rate_sgd "$LEARNING_RATE_SGD" --weight_decay_sgd "$WEIGHT_DECAY_SGD" \
        --learning_rate_adamw "$LEARNING_RATE_ADAMW" --weight_decay_adamw "$WEIGHT_DECAY_ADAMW" \
        --scheduler "CosineAnnealing_Warmup" --criterion "CrossEntropy_Smoothing" \
        --min_lr "$MIN_LR_ModelNet" \
        --workers "$WORKERS_ModelNet" \
        --alpha_beta "$ALPHA_BETA" --ema "$EMA" --rotation_head "$ROTATION_HEAD"
        echo "====================================================================="


        python cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" \
        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
        --batch_size 50 --epoch "$EPOCH_ModelNet" \
        --optimizer "adamw" \
        --learning_rate_sgd "$LEARNING_RATE_SGD" --weight_decay_sgd "$WEIGHT_DECAY_SGD" \
        --learning_rate_adamw "$LEARNING_RATE_ADAMW" --weight_decay_adamw "$WEIGHT_DECAY_ADAMW" \
        --scheduler "CosineAnnealing_Warmup" --criterion "CrossEntropy_Smoothing" \
        --min_lr "$MIN_LR_ModelNet" \
        --workers "$WORKERS_ModelNet" \
        --alpha_beta "$ALPHA_BETA" --ema "yes" --rotation_head "$ROTATION_HEAD"
        echo "====================================================================="

done


#for i in {1..100}; do
#        for ROTATION_HEAD in "${ROTATION_HEAD[@]}"; do
#                for NORM_MODE in "${NORM_MODE[@]}"; do
#                        python cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
#                        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
#                        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" \
#                        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
#                        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
#                        --batch_size "$BATCH_SIZE_ModelNet" --epoch "$EPOCH_ModelNet" \
#                        --optimizer "$OPTIMIZER" \
#                        --learning_rate_sgd "$LEARNING_RATE_SGD" --weight_decay_sgd "$WEIGHT_DECAY_SGD" \
#                        --learning_rate_adamw "$LEARNING_RATE_ADAMW" --weight_decay_adamw "$WEIGHT_DECAY_ADAMW" \
#                        --scheduler "$SCHEDULER" --criterion "$CRITERION" \
#                        --learning_rate "$LEARNING_RATE_ModelNet" --min_lr "$MIN_LR_ModelNet" \
#                        --weight_decay "$WEIGHT_DECAY_ModelNet" --workers "$WORKERS_ModelNet" \
#                        --alpha_beta "$ALPHA_BETA" --ema "$EMA" --rotation_head "$ROTATION_HEAD"
#                        # || { echo "Error in cls_modelnet.py"; exit 1; } --seed "$SEED"
#                        echo "====================================================================="
#                        # --use_xyz True --bias False --transfer_mode "$TRANSFER_MODE" 
#                done
#        done
#done

