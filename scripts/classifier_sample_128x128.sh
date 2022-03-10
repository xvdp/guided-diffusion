#!/bin/bash
#
# README calls differs from 64x64, 256x256 & 512x512 in 
# --num_heads 4 instead of --num_head_channels 64
# no --num_heads defaults to 4
# no --num_head_channels defaults to -1
# produces ok samples with this format, other scirpts do not
#
# --classifier_scale 0.5 instead of --classifier_scale 1
# classifier_scale is not so importnant 
#
# can be called with -b -s -t flags
# as per README dowload pretrain to ../models


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
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path ../models/128x128_classifier.pt --model_path ../models/128x128_diffusion.pt $SAMPLE_FLAGS