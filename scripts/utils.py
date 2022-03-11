
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def store_file(npzfile, batch_size, image_size, suffix=""):

    sqr = int(np.sqrt(batch_size))
    batch_size = sqr**2
    ar = np.load(npzfile)
    im = ar.f.arr_0
    classes = ar.f.arr_1
    image = np.vstack([np.hstack(im[i:i+sqr]) for i in range(0,batch_size,sqr)] )
    os.makedirs('../results', exist_ok=True)
    name = os.path.join(os.path.abspath('../results'), f'classifier_sample_{image_size}x{image_size}{suffix}.png')
    Image.fromarray(image).save(name)
    plt.imshow(image)
    plt.show()