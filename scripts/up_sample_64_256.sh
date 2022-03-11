#!/bin/bash
# Running example from README, from scripts filder (slight modification to for classifier and model path ../ )
#
# generates num_samples, on batches of size batch_size
# saves .npz to /tmp
# 
# requires classifier and diffusion models
#
# SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
UPSAMPLER='../models/64_256_upsampler.pt'
BASESAMPLE='../results/samples_4x64x64x3_2/samples_4x64x64x3.npz'

if [ ! -f "$UPSAMPLER" ]; then
    echo "$UPSAMPLER does not exist ! download or change path"
    exit 1
fi
if [ ! -f "$BASESAMPLE" ]; then
    echo "$BASESAMPLE does not exist, generate npz from image size(B,64,64,3)"
    exit 1
fi


BATCH_SIZE=4
SAMPLES=4
TIMESTEP_RESPACE=250

while getopts 'b:s:t:' OPTION; do
  case "$OPTION" in
    b) BATCH_SIZE="${OPTARG}";;
    s)SAMPLES="$OPTARG";;
    t)TIMESTEP_RESPACE="$OPTARG";;
  esac
done

SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $SAMPLES --timestep_respacing $TIMESTEP_RESPACE"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS --model_path $UPSAMPLER --base_samples $BASESAMPLE $SAMPLE_FLAGS