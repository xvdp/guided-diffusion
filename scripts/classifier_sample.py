"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from pprint import pprint

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main(**kwargs):
    args = create_argparser().parse_args()
    kw = {k:v for k,v in kwargs.items() if k in args}

    args.__dict__.update(**kw)


    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    pprint(args_to_dict(args, model_and_diffusion_defaults().keys()))

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f"loading state dict {args.model_path} -> model")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    print(f"loading state dict {args.classifier_path}")
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            # print(f" cond_fn(x,t,y) t {t}")
            # t, batch_size time sampling 999 -> 0 
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            # print(f"  .. x:{x_in.shape}, t:{t.shape}, logits:{logits.shape}, y: {y.shape}")
            # x:(batch_size, channels, image_size, image_size), t:(batch_size), (batch_size, numclasses), y: (batch_size)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            # print(f" .. selected, softmax(logits)[range(), y] {selected}")  #  (batch_size) floats
            cond =  th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            # print(f" .. cond: {tuple(cond.shape)}, args.classifier_scale {args.classifier_scale}")
            # cond: (batch_size, 3, image_size, image_size), args.classifier_scale 0.5

            return cond
            

    def logt(x):
        if isinstance(x, th.Tensor):
            out = f" {tuple(x.shape)}"
            if x.ndim == 1:
                out += f"{x}"
        elif isinstance(x, (int,float)):
            out = f" {x}"
        return out

    def model_fn(x, t, y=None):
        assert y is not None
        print(f"timestep {t.tolist()} conditional y {y.tolist()}")
        #print(f"model_fn, x {logt(x)}, t {logt(t)} y {logt(y)}")
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        if args.use_ddim:
            print("sample_fn = diffusion.ddim_sample_loop: args.use_ddim")
        else: 
            print("sample_fn = diffusion.p_sample_loop: not args.use_ddim")
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # print(f"sample_fn args.batch_size {args.batch_size}, args.image_size {args.image_size} args.clip_denoised {args.clip_denoised} model_kwargs {model_kwargs}")
        # model_kwargs['y']: class conditioner e.g [ 53,  37, 609, 498, 679,  38, 242, 705, 253, 822, 721, 762,  64,  42, 337, 483]
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

"""
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS --model_path models/64_256_upsampler.pt --base_samples 64_samples.npz $SAMPLE_FLAGS



sample_args = {'batch_size': 4, 'num_samples': 100, 'timestep_respacing': 'ls'}

# model_args = {'attention_resolutions':[32,16,8], 'class_cond': True, 'diffusion_steps': 1000,
model_args = {'attention_resolutions':'32,16,8', 'class_cond': True, 'diffusion_steps': 1000,
              'large_size': 256, 'small_size': 64, 'learn_sigma': True, 'noise_schedule': 'linear',
              'num_channels': 192, 'num_heads': 4, 'num_res_blocks': 2, 'resblock_updown': True,
              'use_fp16': True, 'use_scale_shift_norm': True,
              'model_path': 'models/64_256_upsampler.pt', 'base_samples': '64_samples.npz'}

args.__dict__.update(**sample_args)
args.__dict__.update(**model_args)


create_model_and_diffusion 
->


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sample = "/tmp/openai-2022-03-09-12-41-00-464718/samples_100x64x64x3.npz"
batch_size=100
image_size = 64
sample = "/tmp/openai-2022-03-09-13-47-12-741651/samples_16x128x128x3.npz"
batch_size = 16
image_size = 128

sqr = int(np.sqrt(batch_size))
ar = np.load(sample)
im = ar.f.arr_0
classes = ar.f.arr_1
image = np.vstack([np.hstack(im[i:i+sqr]) for i in range(0,batch_size,sqr)] )
os.makedirs('../results', exist_ok=True)
name = os.path.join(os.path.abspath('../results'), f'classifier_sample_{image_size}x{image_size}.png')
Image.fromarray(image).save(name)
plt.imshow(image);plt.show()
"""
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


"""

sample = sample_fn(
    model_fn,
    (args.batch_size, 3, args.image_size, args.image_size),
    clip_denoised=args.clip_denoised,
    model_kwargs=model_kwargs,
    cond_fn=cond_fn,
    device=dist_util.dev(),
)


sample_fn()
diffusion.p_sample_loop(self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    )

    diffusion.p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
        ):

        diffusion.p_sample(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
        )


"""