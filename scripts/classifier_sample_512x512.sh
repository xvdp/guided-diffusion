#!/bin/bash
#
# classifier_scale is not so importnant 
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path ../models/512x512_classifier.pt --model_path ../models/512x512_diffusion.pt $SAMPLE_FLAGS
#
# can be called with -b -s -t flags
# as per README dowload pretrain to ../models


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
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 4.0 --classifier_path ../models/512x512_classifier.pt --model_path ../models/512x512_diffusion.pt $SAMPLE_FLAGS



