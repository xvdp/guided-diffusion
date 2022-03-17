
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
