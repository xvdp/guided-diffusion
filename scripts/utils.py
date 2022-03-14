
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
    batch_size = sqr**2

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

Gaussian Diffusion __init__()
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
 'timestep_respacing': '250',
 'use_checkpoint': False,
 'use_ddim': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}

UNet __init__()
Logging to /tmp/openai-2022-03-14-14-51-17-407538
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
 'timestep_respacing': '250',
 'use_checkpoint': False,
 'use_fp16': True,
 'use_kl': False,
 'use_scale_shift_norm': True}
Unet: 6
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
loading data...
creating samples...

 GaussianDiffusion.p_sample_loop_progressive(
     indices: [249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
 
 # conditional forward 
 SuperResModel.forward(x): torch.Size([2, 3, 256, 256]) low_res torch.Size([2, 3, 64, 64]), t: tensor([999, 999], device='cuda:0'))

/home/z/miniconda3/envs/abj/lib/python3.9/site-packages/torch/nn/functional.py:3631: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  warnings.warn(

 SuperResModel.forward; cat (x, upsampled): torch.Size([2, 6, 256, 256]))

# one step through Unet
 Unet.forward(x: torch.Size([2, 6, 256, 256]), t: torch.Size([2]), y: torch.Size([2]))
    Unet.forward: input.blocks(h: torch.Size([2, 6, 256, 256]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 192, 64, 64]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 384, 16, 16]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768])
    Unet.forward: input.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768])
        Unet.forward: middle.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768])
    Unet.forward: output.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 8, 8]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 16, 16]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 768, 32, 32]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 32, 32]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 64, 64]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 384, 128, 128]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 192, 128, 128]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]))
    Unet.forward: output.blocks(h: torch.Size([2, 192, 256, 256]), emb torch.Size([2, 768]))
    Unet.forward: out(h: torch.Size([2, 192, 256, 256]))

 GaussianDiffusion._predict_xstart_from_eps(x_t: torch.Size([2, 3, 256, 256]),  eps: torch.Size([2, 3, 256, 256])
 GaussianDiffusion.q_posterior_mean_variance(
    posterior_mean:torch.Size([2, 3, 256, 256]),
    posterior_variance: torch.Size([2, 3, 256, 256]),
    posterior_log_variance_clipped: torch.Size([2, 3, 256, 256])
 GaussianDiffusion.p_mean_variance(
    model_mean: torch.Size([2, 3, 256, 256]),
    model_variance: torch.Size([2, 3, 256, 256]),
    model_log_variance: torch.Size([2, 3, 256, 256]),
    pred_xstart: torch.Size([2, 3, 256, 256])
 GaussianDiffusion.p_sample(
    x: torch.Size([2, 3, 256, 256]),
    cond_fn: None,
    sample: torch.Size([2, 3, 256, 256]),
    pred_xstart: torch.Size([2, 3, 256, 256])
 GaussianDiffusion.p_sample_loop(final['sample']: torch.Size([2, 3, 256, 256])
... 249 more loops ...
created 2 samples
"""