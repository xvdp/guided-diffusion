# Running example from README, from scripts filder (slight modification to for classifier and model path ../ )
#
# generates num_samples, on batches of size batch_size
# saves .npz to /tmp
# 
# requires classifier and diffusion models
#
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path ../models/64x64_classifier.pt --classifier_depth 4 --model_path ../models/64x64_diffusion.pt $SAMPLE_FLAGS


# $ python
# import os
# import numpy as np
# import matplotlib.pyplot
# from PIL import Image
#
# ar = np.load('/tmp/openai-2022-03-09-12-41-00-464718/samples_100x64x64x3.npz')
# im = ar.f.arr_0
# classes = ar.f.arr_1
# image = np.vstack([np.hstack(im[i:i+10]) for i in range(0,100,10)] )
# os.makedirs('../results', exist_ok=True)
# name = os.path.join(os.path.abspath('../results'), 'classifier_sample_64x64.png')
# Image.fromarray(image).save(name)
# plt.imshow(image);plt.show()