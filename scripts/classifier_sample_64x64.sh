#!/bin/bash
# Running example from README, from scripts filder (slight modification to for classifier and model path ../ )
#
# generates num_samples, on batches of size batch_size
# saves .npz to /tmp
# 
# requires classifier and diffusion models
#
# SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
CLASSIFIER='../models/64x64_classifier.pt'
DIFFUSION='../models/64x64_diffusion.pt'
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

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path $CLASSIFIER --classifier_depth 4 --model_path $DIFFUSION $SAMPLE_FLAGS
