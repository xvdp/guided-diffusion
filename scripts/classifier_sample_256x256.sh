#!/bin/bash
#
# img 256, batch 16, maxes Titan GPU: 24GB
# SAMPLE_FLAGS="--batch_size 16 --num_samples 16 --timestep_respacing 250"
#
# can be called with -b -s -t flags
# as per README dowload pretrain to ../models

CLASSIFIER='../models/256x256_classifier.pt'
DIFFUSION='../models/256x256_diffusion.pt'
FILES=($CLASSIFIER  $DIFFUSION)
STOP=0
for name in ${FILES[*]}; do
    if [ ! -f "$name" ]; then
        echo "$name does not exist ! download or change path"
        STOP=1
    fi
    if [ $STOP == 1 ]; then
        exit 1
    fi
done

BATCH_SIZE=16
SAMPLES=16
TIMESTEP_RESPACE=250


while getopts 'b:s:t:' OPTION; do
  case "$OPTION" in
    b) BATCH_SIZE="${OPTARG}";;
    s)SAMPLES="$OPTARG";;
    t)TIMESTEP_RESPACE="$OPTARG";;
  esac
done


SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $SAMPLES --timestep_respacing $TIMESTEP_RESPACE"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path $CLASSIFIER --model_path $DIFFUSION $SAMPLE_FLAGS
