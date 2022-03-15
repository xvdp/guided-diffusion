#!/bin/bash
# Running example from README, from scripts filder (slight modification to for classifier and model path ../ )
#
# generates num_samples, on batches of size batch_size
# saves .npz to /tmp
# 
# requires classifier and diffusion models
# 
# SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
#
UPSAMPLER='../models/128_512_upsampler.pt'
BASESAMPLE="/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upsample_downscale_128.npz"
SAMPLES=4
BASESAMPLE='/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upscale128_hare.npz'
BATCH_SIZE=2
SAMPLES=2

if [ ! -f "$UPSAMPLER" ]; then
    echo "'$UPSAMPLER' does not exist ! download or change path"
    exit 1
fi
if [ ! -f "$BASESAMPLE" ]; then
    echo "'$BASESAMPLE' does not exist, generate npz from image size(B,128,128,3)"
    exit 1
fi

TIMESTEP_RESPACE=250

while getopts 'b:s:t:' OPTION; do
  case "$OPTION" in
    b) BATCH_SIZE="${OPTARG}";;
    s)SAMPLES="$OPTARG";;
    t)TIMESTEP_RESPACE="$OPTARG";;
  esac
done

SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $SAMPLES --timestep_respacing $TIMESTEP_RESPACE"

MODEL_FLAGS="--attention_resolutions 32,16 --class_cond True --diffusion_steps 1000 --large_size 512 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python x_super_res_sample.py $MODEL_FLAGS --model_path $UPSAMPLER $SAMPLE_FLAGS --base_samples $BASESAMPLE