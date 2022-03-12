# Guided Diffusion Observations

Scripts from README.md have been translated to bash files inside `scripts/`

# Upsampling
`scripts/up_sample_64_256.sh`

Trained upscaling model is conditional. Running it with an .npz without classes fails lo load on account of missing parameter in model `'label_emb.weight'`. Loading data without class conditional fails to upscale.

To run upsampling either:
* generate 64x64 samples with `scripts/classifier_sample_64x64.sh` then use the generated .npz.
* or create an .npz with 2 arrays `.arr_0` with (b,3,64,64) uint8 data and `.arr_1` (b,) int classes; rough script to generate npz from .jpg or .png files in `scripts/utils.py`

## Observations
Even though generated data fails to capture larger scale structure information, textural quality of upsampling is higher when run over generated data than data from the wild; this is true even when `classes` match ImageNet classes.  `Does this mean that the learned model is biased towards ImageNet surface details?` 

### Upsample images in the wild
<div align="center">
    <table><tr>
        <td> <img width="100%" src=".github/upsample_original_.png"/> <figcaption>From The Wild 256x256 patches</figcaption></td>
        <td> <img width="100%" src=".github/upsample_downscale_.png"/><figcaption>Downscaled 64x64</figcaption></td>
        <td> <img width="100%" src=".github/upsample_upsampled_.png"/><figcaption>Up sampled to 256x256</figcaption></td>
    </tr></table>
</div>

### Upsample Generated Data
<div align="center">
    <table><tr>
        <!-- <td> <img width="100%" src=".github/upsample_original_.png"/> <figcaption>Original 256x256 patches</figcaption></td> -->
        <td> <img width="60%" src=".github/classifier_sample_16x64x64.png"/><figcaption>Generated 64x64</figcaption></td>
        <td> <img width="60%" src=".github/upsample_generated_256x256.png"/><figcaption>Up sampled to 256x256</figcaption></td>
    </tr></table>
</div>
<!-- [12, 863, 862, 201, 337, 788, 287, 857, 98, 50, 138, 445, 545, 330, 469, 534] -->
<!-- ['house_finch', 'totem_pole', 'torch', 'silky_terrier', 'beaver', 'shoe_shop', 'lynx', 'throne', 'red-breasted_merganser', 'American_alligator', 'bustard', 'bikini', 'electric_fan', 'wood_rabbit', 'caldron', 'dishwasher'] -->