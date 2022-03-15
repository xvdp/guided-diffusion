"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
Simple shortcircuiting kwarg indirection
"""

import argparse
import os
from pprint import pprint

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# pylint: disable=no-member
def super_res():
    """ inspect only - doesnt round model
    """
    base_samples = '../results/samples_4x64x64x3_2/samples_4x64x64x3.npz'
    class_cond = True
    base_samples = '/home/z/work/gits/Diffusion/guided-diffusion/results/upscale64/upsample_downscale_.npz'
    class_cond = False

    sample_args = {
        'base_samples': base_samples,
        'batch_size': 4,
        'clip_denoised': True,
        'model_path': '../models/64_256_upsampler.pt',
        'num_samples': 16,
        'use_ddim': False}

    args = {'attention_resolutions': '32,16,8',
            'class_cond': class_cond,
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
            'use_scale_shift_norm': True,
            }


    model, diffusion = sr_create_model_and_diffusion(**args)
    return model, sample_args['model_path']

def load_tolerant(model, checkpoint):
    state_dict = th.load(checkpoint, map_location="cpu")
    model_state_dict = model.state_dict()

    not_in_model = [k for k in state_dict if k not in model_state_dict]
    not_in_checkpoint = [k for k in model_state_dict if k not in state_dict]
    matched = [k for k in model_state_dict if k in state_dict]
    for k in matched:
       model_state_dict[k] = state_dict[k]
    model.load_state_dict(model_state_dict)
    return matched, not_in_model, not_in_checkpoint

def main():
    args = create_argparser().parse_args()
    pprint({k:v for k,v in args.__dict__.items()})

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    pprint(args_to_dict(args, sr_model_and_diffusion_defaults().keys()))

    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    # skips 
    # load_tolerant(model, args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
