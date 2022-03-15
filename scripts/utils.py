
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image



def npz_png(npzfile, prefix="classifier_sample", suffix="", folder='../results'):
    """ converts npz to png; only powers of 2
    eg. if npz shape == 6,256,256,3, only first 4 items will be stored
    """

    ar = np.load(npzfile)
    im = ar[ar.files[0]]

    batch_size, image_size, *_ = im.shape
    sqr = int(np.sqrt(batch_size))
    batch_size = sqr**2 if sqr > 1 else batch_size

    # classes = ar.f.arr_1
    image = np.vstack([np.hstack(im[i:i+sqr]) for i in range(0, batch_size, sqr)] )
    os.makedirs(folder, exist_ok=True)
    name = os.path.join(os.path.abspath(folder), f'{prefix}_{image_size}x{image_size}{suffix}.png')
    Image.fromarray(image).save(name)
    plt.imshow(image)
    plt.show()


def center_crop(im, size, rand=False):
    w, h = im.size
    assert w >= size and h >= size, f"image is too small{im.size} > {size}"
    if rand:
        left = np.random.randint(0,w - size)
        upper = np.random.randint(0,h - size)
    else:
        left = (w - size)//2
        upper = (h- size)//2
    return im.crop((left, upper, left+size, upper+size))

def images_npz(files, downscale=4, cropsize=64, prefix="upsample", suffix="", folder='../results', show=True):
    """ list of images:
        1. to .png scaled or cropped
        2. to .png downscaled (1.) by factor 'downscale'
        3. to .npz downscaled (1.) by factor 'downscale'

    Example
    >>> where = '/home/z/data/Self/Animals'
    >>> files = [f.path for f in os.scandir(where) if f.name[-4:].lower() in (".jpg", ".png") ]
    >>> images_npz(files[:7]+files[8:9], folder="../results/upscale64")
    """
    num = len(files)
    sqr = int(np.sqrt(num * 2))
    num = max(1, sqr**2 //2)
    uncropped = []
    original=[]
    down=[]
    for i in range(num):
        im = Image.open(files[i]).convert("RGB")
        uncropped.append(im)
        #
        scale = min(im.size) / (cropsize*downscale)
        nusize = tuple([int(s//scale) for s in im.size])
        imscaled = im.resize(nusize)
        #
        # large fov original resolution crop
        imscaled = center_crop(imscaled, cropsize*downscale)
        original.append(np.asarray(imscaled))
        # large fov downscaled
        nusize = tuple([s//downscale for s in imscaled.size])
        down.append(np.asarray(imscaled.resize(nusize)))
        #
        # crop of original resolution
        im = center_crop(im, cropsize*downscale)
        original.append(np.asarray(im))
        # crop of downscaled
        nusize = tuple([s//downscale for s in im.size])
        down.append(np.asarray(im.resize(nusize)))
    # numpy
    original = np.stack(original)
    down = np.stack(down)
    # gridded numpy for png
    ori_image = np.vstack([np.hstack(original[i:i+sqr]) for i in range(0, num*2, sqr)] )
    ori_down = np.vstack([np.hstack(down[i:i+sqr]) for i in range(0, num*2, sqr)] )

    os.makedirs(folder, exist_ok=True)
    name = os.path.join(os.path.abspath(folder), f'{prefix}_original_{suffix}.png')
    Image.fromarray(ori_image).save(name)
    name = os.path.join(os.path.abspath(folder), f'{prefix}_downscale_{suffix}')
    Image.fromarray(ori_down).save(name+'.png')
    name = os.path.join(os.path.abspath(folder), f'{prefix}_downscale_{suffix}.npz')
    np.savez(name, arr_0=down)
    print(f"saved {name}..")

    if show:
        plt.subplot(1,2,1)
        plt.imshow(ori_image)
        plt.subplot(1,2,2)
        plt.imshow(ori_down)
        plt.show()

def add_class_npz(npz, classlist):
    """ add approx classes to test upsampling
    npz='/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upsample_downscale__1.npz'
    classlist = [377,377,377,377,296,296,56,56,277,277,56,56,277,277,279,279]
    add_class_npz(npz, classlist)

    """
    data = np.load(npz)
    # if len(data.files) == 2:
    #     print("file contains classes already", )
    #     return

    assert len(data[data.files[0]]) == len(classlist), f"expected {len(data[data.files[0]])} classes, got {len(classlist)}"
    np.savez(npz, arr_0=data[data.files[0]], arr_1=np.asarray(classlist))
    print(f"saved file {npz} with classes")

def class_name(arr):
    with open("classes.txt", 'r') as fi:
        txt = fi.read().split("\n")
    return [txt[i] for i in arr]


"""Upsampming model trace input [2,64,64,3] -> [2,256,256,3]

{'attention_resolutions': '32,16,8',
 'base_samples': '/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upsample_hare_downscale_.npz',
 'batch_size': 2,
 'class_cond': True,
 'clip_denoised': True,
 'diffusion_steps': 1000,
 'dropout': 0.0,
 'large_size': 256,
 'learn_sigma': True,
 'model_path': '../models/64_256_upsampler.pt',
 'noise_schedule': 'linear',
 'num_channels': 192,
 'num_head_channels': -1,
 'num_heads': 4,
 'num_heads_upsample': -1,
 'num_res_blocks': 2,
 'num_samples': 2,
 'predict_xstart': False,
 'resblock_updown': True,
 'rescale_learned_sigmas': False,
 'rescale_timesteps': False,
 'small_size': 64,
 'timestep_respacing': '1000',
 'use_checkpoint': False,
 'use_ddim': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}
Logging to /tmp/openai-2022-03-15-12-46-17-071707
creating model...
{'attention_resolutions': '32,16,8',
 'class_cond': True,
 'diffusion_steps': 1000,
 'dropout': 0.0,
 'large_size': 256,
 'learn_sigma': True,
 'noise_schedule': 'linear',
 'num_channels': 192,
 'num_head_channels': -1,
 'num_heads': 4,
 'num_heads_upsample': -1,
 'num_res_blocks': 2,
 'predict_xstart': False,
 'resblock_updown': True,
 'rescale_learned_sigmas': False,
 'rescale_timesteps': False,
 'small_size': 64,
 'timestep_respacing': '1000',
 'use_checkpoint': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}

SuperResModel.__init__():
 Unet.__init__():
	in_channels 6
	out_channels 6
	model_channels 192
	num_res_blocks 2
	attention_resolutions (8, 16, 32)
	dropout 0.0
	conv_resample True
	num_classes 1000
	num_heads 4
	num_head_channels -1
	num_heads_upsample 4
	level[0], mult[1], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[0] +TimestepEmbedSequential(*layers 1
	level[0], mult[1], res_block[1] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[1] +TimestepEmbedSequential(*layers 1
	level[0], mult[1], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 192, time_embed_dim 768, dropout 0.0, out_channels 192, dims: 2
	 level[1], mult[1], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[0] +TimestepEmbedSequential(*layers 1
	 level[1], mult[1], res_block[1] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[1] +TimestepEmbedSequential(*layers 1
	 level[1], mult[1], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 192, time_embed_dim 768, dropout 0.0, out_channels 192, dims: 2
	  level[2], mult[2], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[0] +TimestepEmbedSequential(*layers 1
	  level[2], mult[2], res_block[1] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[1] +TimestepEmbedSequential(*layers 1
	  level[2], mult[2], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 384, time_embed_dim 768, dropout 0.0, out_channels 384, dims: 2
	   level[3], mult[2], res_block[0] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	   level[3], mult[2] ds 8 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch384, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	   level[3], mult[2], res_block[0] +TimestepEmbedSequential(*layers 2
	   level[3], mult[2], res_block[1] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	   level[3], mult[2] ds 8 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch384, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	   level[3], mult[2], res_block[1] +TimestepEmbedSequential(*layers 2
	   level[3], mult[2], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 384, time_embed_dim 768, dropout 0.0, out_channels 384, dims: 2
	    level[4], mult[4], res_block[0] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4] ds 16 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	    level[4], mult[4], res_block[0] +TimestepEmbedSequential(*layers 2
	    level[4], mult[4], res_block[1] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4] ds 16 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	    level[4], mult[4], res_block[1] +TimestepEmbedSequential(*layers 2
	    level[4], mult[4], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	     level[5], mult[4], res_block[0] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	     level[5], mult[4] ds 32 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	     level[5], mult[4], res_block[0] +TimestepEmbedSequential(*layers 2
	     level[5], mult[4], res_block[1] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	     level[5], mult[4] ds 32 in attention_resolutions(8, 16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1), use_new_attention_order:False
	     level[5], mult[4], res_block[1] +TimestepEmbedSequential(*layers 2
	      MiddleBlock: +TimestepEmbedSequential(ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	      AttentionBlock (ch: 768, num_heads 4, num_head_channels -1, use_new_attention_order False
	      ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	self._feature_size += ch  8256
	     level[5], mult[4], res_block[0] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[0: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	     level[5], mult[4], res_block[1] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[1: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	     level[5], mult[4], res_block[2] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[2: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	     level[5], mult[4], res_block[2, level and i == num_res_blocks] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4], res_block[0] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[0: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	    level[4], mult[4], res_block[1] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[1: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	    level[4], mult[4], res_block[2] +ResBlock(ch768 +ich384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[2: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:-1)
	    level[4], mult[4], res_block[2, level and i == num_res_blocks] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	   level[3], mult[2], res_block[0] +ResBlock(ch768 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], AttentionBlock[0: ds in attention_resolutions] +AttentionBlock(ch384, num_heads:4, num_head_channels:-1)
	   level[3], mult[2], res_block[1] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], AttentionBlock[1: ds in attention_resolutions] +AttentionBlock(ch384, num_heads:4, num_head_channels:-1)
	   level[3], mult[2], res_block[2] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], AttentionBlock[2: ds in attention_resolutions] +AttentionBlock(ch384, num_heads:4, num_head_channels:-1)
	   level[3], mult[2], res_block[2, level and i == num_res_blocks] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[0] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[1] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[2] +ResBlock(ch384 +ich192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[2, level and i == num_res_blocks] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[0] +ResBlock(ch384 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[1] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[2] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[2, level and i == num_res_blocks] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[0] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	level[0], mult[1], res_block[1] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	level[0], mult[1], res_block[2] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
 GaussianDiffusion.__init__()
 GaussianDiffusion.__init__()
loading data...
creating samples...
 GaussianDiffusion.p_sample_loop_progressive(indices: 1000
 1.GaussianDiffusion.p_sample(x: torch.Size([2, 3, 256, 256]), cond_fn: None, t: tensor([999, 999], device='cuda:0'), model: SuperResModel model_kwargs: {'low_res': torch.Size([2, 3, 64, 64]), 'y': torch.Size([2])}
 0.GaussianDiffusion.p_mean_variance(x: torch.Size([2, 3, 256, 256]))
SuperResModel.forward(x): torch.Size([2, 3, 256, 256]) low_res torch.Size([2, 3, 64, 64]), t: tensor([999, 999], device='cuda:0'))
/home/z/miniconda3/envs/abj/lib/python3.9/site-packages/torch/nn/functional.py:3631: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  warnings.warn(
SuperResModel.forward; x = cat(x, upsampled): torch.Size([2, 6, 256, 256]))
Unet.forward(x: torch.Size([2, 6, 256, 256]), t: torch.Size([2]), y: torch.Size([2]))
Unet.forward(emb: torch.Size([2, 768]), timestep_embedding: torch.Size([2]), channels: 192)
Unet.forward: input.blocks(h: torch.Size([2, 6, 256, 256]), emb torch.Size([2, 768]) -> residual(0)
 Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(1)
  Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(2)
   Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(3)
    Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]) -> residual(4)
     Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]) -> residual(5)
      Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]) -> residual(6)
       Unet.forward: input.blocks(h: torch.Size([2, 192, 64, 64]), emb torch.Size([2, 768]) -> residual(7)
        Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]) -> residual(8)
         Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]) -> residual(9)
          Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]) -> residual(10)
           Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]) -> residual(11)
            Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]) -> residual(12)
             Unet.forward: input.blocks(h: torch.Size([2, 384, 16, 16]), emb torch.Size([2, 768]) -> residual(13)
              Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]) -> residual(14)
               Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]) -> residual(15)
                Unet.forward: input.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768]) -> residual(16)
                 Unet.forward: input.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768]) -> residual(17)
                  Unet.forward: middle.blocks ->(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768])
                  Unet.forward: middle.blocks <-(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768])
                  Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 8, 8]) <- residual.pop(17) emb torch.Size([2, 768]))
                 Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 8, 8]) <- residual.pop(16) emb torch.Size([2, 768]))
                Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 8, 8]) <- residual.pop(15) emb torch.Size([2, 768]))
               Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(14) emb torch.Size([2, 768]))
              Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(13) emb torch.Size([2, 768]))
             Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(12) emb torch.Size([2, 768]))
            Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 32, 32]) <- residual.pop(11) emb torch.Size([2, 768]))
           Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 32, 32]) <- residual.pop(10) emb torch.Size([2, 768]))
          Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 32, 32]) <- residual.pop(9) emb torch.Size([2, 768]))
         Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 64, 64]) <- residual.pop(8) emb torch.Size([2, 768]))
        Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 64, 64]) <- residual.pop(7) emb torch.Size([2, 768]))
       Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 64, 64]) <- residual.pop(6) emb torch.Size([2, 768]))
      Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 128, 128]) <- residual.pop(5) emb torch.Size([2, 768]))
     Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 128, 128]) <- residual.pop(4) emb torch.Size([2, 768]))
    Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 128, 128]) <- residual.pop(3) emb torch.Size([2, 768]))
   Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 256, 256]) <- residual.pop(2) emb torch.Size([2, 768]))
  Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 256, 256]) <- residual.pop(1) emb torch.Size([2, 768]))
 Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 256, 256]) <- residual.pop(0) emb torch.Size([2, 768]))
 Unet.forward: out(h: torch.Size([2, 192, 256, 256]))
Unet.forward: out: torch.Size([2, 6, 256, 256]))
x = SuperResModel.forward(x): torch.Size([2, 6, 256, 256])
 1.GaussianDiffusion.p_mean_variance(model_output: torch.Size([2, 6, 256, 256]), model _WrappedModel)
  START_X GaussianDiffusion.p_mean_variance(model_mean_type: ModelMeanType.EPSILON, x: torch.Size([2, 3, 256, 256]), eps: torch.Size([2, 3, 256, 256])
 GaussianDiffusion._predict_xstart_from_eps(x_t: torch.Size([2, 3, 256, 256]),  eps: torch.Size([2, 3, 256, 256])
 GaussianDiffusion.q_posterior_mean_variance( posterior_mean:torch.Size([2, 3, 256, 256]), posterior_variance: torch.Size([2, 3, 256, 256]), posterior_log_variance_clipped: torch.Size([2, 3, 256, 256])
  START_X GaussianDiffusion.p_mean_variance(model_mean: torch.Size([2, 3, 256, 256])
 6. GaussianDiffusion.p_mean_variance(model_mean: torch.Size([2, 3, 256, 256]), model_variance: torch.Size([2, 3, 256, 256]), model_log_variance: torch.Size([2, 3, 256, 256]), pred_xstart: torch.Size([2, 3, 256, 256])
 2.GaussianDiffusion.p_sample(x: torch.Size([2, 3, 256, 256]), cond_fn: None, sample: torch.Size([2, 3, 256, 256]) pred_xstart: torch.Size([2, 3, 256, 256])
 
 
 ##
 # upsampling from 128-512
 #
 {'attention_resolutions': '32,16',
 'base_samples': '/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upscale128_hare.npz',
 'batch_size': 2,
 'class_cond': True,
 'clip_denoised': True,
 'diffusion_steps': 1000,
 'dropout': 0.0,
 'large_size': 512,
 'learn_sigma': True,
 'model_path': '../models/128_512_upsampler.pt',
 'noise_schedule': 'linear',
 'num_channels': 192,
 'num_head_channels': 64,
 'num_heads': 4,
 'num_heads_upsample': -1,
 'num_res_blocks': 2,
 'num_samples': 2,
 'predict_xstart': False,
 'resblock_updown': True,
 'rescale_learned_sigmas': False,
 'rescale_timesteps': False,
 'small_size': 128,
 'timestep_respacing': '1000',
 'use_checkpoint': False,
 'use_ddim': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}
Logging to /tmp/openai-2022-03-15-13-01-34-134770
creating model...
{'attention_resolutions': '32,16',
 'class_cond': True,
 'diffusion_steps': 1000,
 'dropout': 0.0,
 'large_size': 512,
 'learn_sigma': True,
 'noise_schedule': 'linear',
 'num_channels': 192,
 'num_head_channels': 64,
 'num_heads': 4,
 'num_heads_upsample': -1,
 'num_res_blocks': 2,
 'predict_xstart': False,
 'resblock_updown': True,
 'rescale_learned_sigmas': False,
 'rescale_timesteps': False,
 'small_size': 128,
 'timestep_respacing': '1000',
 'use_checkpoint': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}
SuperResModel.__init__():
Unet.__init__():
	in_channels 6
	out_channels 6
	model_channels 192
	num_res_blocks 2
	attention_resolutions (16, 32)
	dropout 0.0
	conv_resample True
	num_classes 1000
	num_heads 4
	num_head_channels 64
	num_heads_upsample 4
	level[0], mult[1], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[0] +TimestepEmbedSequential(*layers 1
	level[0], mult[1], res_block[1] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[1] +TimestepEmbedSequential(*layers 1
	level[0], mult[1], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 192, time_embed_dim 768, dropout 0.0, out_channels 192, dims: 2
	 level[1], mult[1], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[0] +TimestepEmbedSequential(*layers 1
	 level[1], mult[1], res_block[1] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[1] +TimestepEmbedSequential(*layers 1
	 level[1], mult[1], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 192, time_embed_dim 768, dropout 0.0, out_channels 192, dims: 2
	  level[2], mult[2], res_block[0] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[0] +TimestepEmbedSequential(*layers 1
	  level[2], mult[2], res_block[1] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[1] +TimestepEmbedSequential(*layers 1
	  level[2], mult[2], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 384, time_embed_dim 768, dropout 0.0, out_channels 384, dims: 2
	   level[3], mult[2], res_block[0] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	   level[3], mult[2], res_block[0] +TimestepEmbedSequential(*layers 1
	   level[3], mult[2], res_block[1] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	   level[3], mult[2], res_block[1] +TimestepEmbedSequential(*layers 1
	   level[3], mult[2], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 384, time_embed_dim 768, dropout 0.0, out_channels 384, dims: 2
	    level[4], mult[4], res_block[0] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4] ds 16 in attention_resolutions(16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:64), use_new_attention_order:False
	    level[4], mult[4], res_block[0] +TimestepEmbedSequential(*layers 2
	    level[4], mult[4], res_block[1] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4] ds 16 in attention_resolutions(16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:64), use_new_attention_order:False
	    level[4], mult[4], res_block[1] +TimestepEmbedSequential(*layers 2
	    level[4], mult[4], res_block[1] level != len(channel_mult) - 1] +TimestepEmbedSequential(ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	     level[5], mult[4], res_block[0] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	     level[5], mult[4] ds 32 in attention_resolutions(16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:64), use_new_attention_order:False
	     level[5], mult[4], res_block[0] +TimestepEmbedSequential(*layers 2
	     level[5], mult[4], res_block[1] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	     level[5], mult[4] ds 32 in attention_resolutions(16, 32)] +AttentionBlock(ch768, num_heads:4, num_head_channels:64), use_new_attention_order:False
	     level[5], mult[4], res_block[1] +TimestepEmbedSequential(*layers 2
	      MiddleBlock: +TimestepEmbedSequential(ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	      AttentionBlock (ch: 768, num_heads 4, num_head_channels 64, use_new_attention_order False
	      ResBlock (ch: 768, time_embed_dim 768, dropout 0.0, out_channels 768, dims: 2
	self._feature_size += ch  8256
	     level[5], mult[4], res_block[0] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[0: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	     level[5], mult[4], res_block[1] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[1: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	     level[5], mult[4], res_block[2] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	     level[5], mult[4], AttentionBlock[2: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	     level[5], mult[4], res_block[2, level and i == num_res_blocks] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	    level[4], mult[4], res_block[0] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[0: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	    level[4], mult[4], res_block[1] +ResBlock(ch768 +ich768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[1: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	    level[4], mult[4], res_block[2] +ResBlock(ch768 +ich384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True, UP
	    level[4], mult[4], AttentionBlock[2: ds in attention_resolutions] +AttentionBlock(ch768, num_heads:4, num_head_channels:64)
	    level[4], mult[4], res_block[2, level and i == num_res_blocks] +ResBlock(ch768, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	   level[3], mult[2], res_block[0] +ResBlock(ch768 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], res_block[1] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], res_block[2] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	   level[3], mult[2], res_block[2, level and i == num_res_blocks] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:768, dims:2, use_scale_shift_norm:True
	  level[2], mult[2], res_block[0] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[1] +ResBlock(ch384 +ich384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[2] +ResBlock(ch384 +ich192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True, UP
	  level[2], mult[2], res_block[2, level and i == num_res_blocks] +ResBlock(ch384, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	 level[1], mult[1], res_block[0] +ResBlock(ch384 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[1] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[2] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	 level[1], mult[1], res_block[2, level and i == num_res_blocks] +ResBlock(ch192, time_embed_dim:768, dropout:0.0), out_channels:384, dims:2, use_scale_shift_norm:True
	level[0], mult[1], res_block[0] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	level[0], mult[1], res_block[1] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
	level[0], mult[1], res_block[2] +ResBlock(ch192 +ich192, time_embed_dim:768, dropout:0.0), out_channels:192, dims:2, use_scale_shift_norm:True, UP
 GaussianDiffusion.__init__()
 GaussianDiffusion.__init__()
loading data...
creating samples...
 GaussianDiffusion.p_sample_loop_progressive(indices: 1000
 1.GaussianDiffusion.p_sample(x: torch.Size([2, 3, 512, 512]), cond_fn: None, t: tensor([999, 999], device='cuda:0'), model: SuperResModel model_kwargs: {'low_res': torch.Size([2, 3, 128, 128]), 'y': torch.Size([2])}
 0.GaussianDiffusion.p_mean_variance(x: torch.Size([2, 3, 512, 512]))
SuperResModel.forward(x): torch.Size([2, 3, 512, 512]) low_res torch.Size([2, 3, 128, 128]), t: tensor([999, 999], device='cuda:0'))
/home/z/miniconda3/envs/abj/lib/python3.9/site-packages/torch/nn/functional.py:3631: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  warnings.warn(
SuperResModel.forward; x = cat(x, upsampled): torch.Size([2, 6, 512, 512]))
Unet.forward(x: torch.Size([2, 6, 512, 512]), t: torch.Size([2]), y: torch.Size([2]))
Unet.forward(emb: torch.Size([2, 768]), timestep_embedding: torch.Size([2]), channels: 192)
Unet.forward: input.blocks(h: torch.Size([2, 6, 512, 512]), emb torch.Size([2, 768]) -> residual(0)
 Unet.forward: input.blocks(h: torch.Size([2, 192, 512, 512]), emb torch.Size([2, 768]) -> residual(1)
  Unet.forward: input.blocks(h: torch.Size([2, 192, 512, 512]), emb torch.Size([2, 768]) -> residual(2)
   Unet.forward: input.blocks(h: torch.Size([2, 192, 512, 512]), emb torch.Size([2, 768]) -> residual(3)
    Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(4)
     Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(5)
      Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]) -> residual(6)
       Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]) -> residual(7)
        Unet.forward: input.blocks(h: torch.Size([2, 384, 128, 128]), emb torch.Size([2, 768]) -> residual(8)
         Unet.forward: input.blocks(h: torch.Size([2, 384, 128, 128]), emb torch.Size([2, 768]) -> residual(9)
          Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]) -> residual(10)
           Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]) -> residual(11)
            Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]) -> residual(12)
             Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]) -> residual(13)
              Unet.forward: input.blocks(h: torch.Size([2, 768, 32, 32]), emb torch.Size([2, 768]) -> residual(14)
               Unet.forward: input.blocks(h: torch.Size([2, 768, 32, 32]), emb torch.Size([2, 768]) -> residual(15)
                Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]) -> residual(16)
                 Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]) -> residual(17)
                  Unet.forward: middle.blocks ->(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768])
                  Unet.forward: middle.blocks <-(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768])
                  Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(17) emb torch.Size([2, 768]))
                 Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(16) emb torch.Size([2, 768]))
                Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 16, 16]) <- residual.pop(15) emb torch.Size([2, 768]))
               Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 32, 32]) <- residual.pop(14) emb torch.Size([2, 768]))
              Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 32, 32]) <- residual.pop(13) emb torch.Size([2, 768]))
             Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 32, 32]) <- residual.pop(12) emb torch.Size([2, 768]))
            Unet.forward: output.blocks(cat(h: torch.Size([2, 768, 64, 64]) <- residual.pop(11) emb torch.Size([2, 768]))
           Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 64, 64]) <- residual.pop(10) emb torch.Size([2, 768]))
          Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 64, 64]) <- residual.pop(9) emb torch.Size([2, 768]))
         Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 128, 128]) <- residual.pop(8) emb torch.Size([2, 768]))
        Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 128, 128]) <- residual.pop(7) emb torch.Size([2, 768]))
       Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 128, 128]) <- residual.pop(6) emb torch.Size([2, 768]))
      Unet.forward: output.blocks(cat(h: torch.Size([2, 384, 256, 256]) <- residual.pop(5) emb torch.Size([2, 768]))
     Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 256, 256]) <- residual.pop(4) emb torch.Size([2, 768]))
    Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 256, 256]) <- residual.pop(3) emb torch.Size([2, 768]))
   Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 512, 512]) <- residual.pop(2) emb torch.Size([2, 768]))
  Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 512, 512]) <- residual.pop(1) emb torch.Size([2, 768]))
 Unet.forward: output.blocks(cat(h: torch.Size([2, 192, 512, 512]) <- residual.pop(0) emb torch.Size([2, 768]))
 Unet.forward: out(h: torch.Size([2, 192, 512, 512]))
Unet.forward: out: torch.Size([2, 6, 512, 512]))
x = SuperResModel.forward(x): torch.Size([2, 6, 512, 512])
 1.GaussianDiffusion.p_mean_variance(model_output: torch.Size([2, 6, 512, 512]), model _WrappedModel)
  START_X GaussianDiffusion.p_mean_variance(model_mean_type: ModelMeanType.EPSILON, x: torch.Size([2, 3, 512, 512]), eps: torch.Size([2, 3, 512, 512])
 GaussianDiffusion._predict_xstart_from_eps(x_t: torch.Size([2, 3, 512, 512]),  eps: torch.Size([2, 3, 512, 512])
 GaussianDiffusion.q_posterior_mean_variance( posterior_mean:torch.Size([2, 3, 512, 512]), posterior_variance: torch.Size([2, 3, 512, 512]), posterior_log_variance_clipped: torch.Size([2, 3, 512, 512])
  START_X GaussianDiffusion.p_mean_variance(model_mean: torch.Size([2, 3, 512, 512])
 6. GaussianDiffusion.p_mean_variance(model_mean: torch.Size([2, 3, 512, 512]), model_variance: torch.Size([2, 3, 512, 512]), model_log_variance: torch.Size([2, 3, 512, 512]), pred_xstart: torch.Size([2, 3, 512, 512])
 2.GaussianDiffusion.p_sample(x: torch.Size([2, 3, 512, 512]), cond_fn: None, sample: torch.Size([2, 3, 512, 512]) pred_xstart: torch.Size([2, 3, 512, 512])

 """