import os
import numpy as np
import torch as th
from PIL import Image, ImageDraw

_DEBUG = False
def print_cond(*msg, cond=_DEBUG):
    if cond:
        print(*msg)

def logtensor(tensor, what=('shape', 'grad', 'mean', 'std', 'minmax')):
    out=""
    if "shape" in what:
        out += f"{tuple(tensor.shape)}"
    other = len(what) > 1
    if other:
        out +="["
    if "grad" in what:
        out += f"∇:{tensor.requires_grad}"
    if tensor.is_floating_point():
        if "mean" in what:
            out += f" μ:{tensor.mean().item():.2f}"
        if "std" in what:
            out += f" σ:{tensor.std().item():.2f}"
        if "minmax" in what:
            out += f" R:[{tensor.min().item():.2f}:{tensor.max().item():.2f}"
    if other:
        out +="]"
    return out

def logkwargs(kwargs):
    out = ""
    if kwargs:
        out = "{"
        for key, val in kwargs.items():
            out += f"{key}:" 
            if isinstance(val, th.Tensor):
                out += logtensor(val)
            else:
                out += f"{val}"
            out +=","
        out +="}"
    return out

# pylint: disable=no-member
def save_image(*tensors, name="super_fwd", i=0, folder="/home/z/temp/guided_sample", names=None):
    os.makedirs(folder, exist_ok=True)

    norm = lambda x: (x - x.min())/(x.max() - x.min())*255
    image = [norm(x) for x in [x.cpu().clone().detach() for x in tensors]]
    num = len(image)
    height = image[0].shape[2]
    

    image = th.cat([c for c in th.cat(image, dim=2)], dim=-1).permute(1,2,0).contiguous().numpy()
    im = Image.fromarray(image.astype(np.uint8))
    if names is not None:
        draw = ImageDraw.Draw(im)
        for j, n in enumerate(names):
            if j >= num:
                break
            draw.text((10, 10 + j*height), n)
    name = os.path.join(folder, f"{name}_{i:06d}.png")
    print(f"...saving: {name}\n")


    im.save(name)
