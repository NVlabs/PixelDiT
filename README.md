<p align="center">
  <img src="assets/pixeldit-logo.png" alt="PixelDiT" width="360">
</p>

<h1 align="center"><a href="https://pixeldit.github.io/">PixelDiT </a> &amp;  <a href="https://pixeldit.github.io/pixeldit2/">PixelDiT2 </a> </h1>
<p align="center"><strong>Pixel-space diffusion transformers for image generation</strong></p>
<p align="center">NVIDIA · University of Rochester</p>


![PixelDiT text-to-image samples](assets/pixeldit-t2i.jpg)


## Papers and Resources

| Work | Publication | Resources |
|---|---|---|
| **PixelDiT2: Representation-Grounded Pixel Diffusion Transformers** | **NeurIPS 2026** | <a href="https://pixeldit.github.io/pixeldit2/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_-Project-2ea44f" alt="Project"></a> <a href="https://arxiv.org/abs/2609.24919"><img src="https://img.shields.io/badge/%F0%9F%93%84_-arXiv-b31b1b.svg" alt="arXiv"></a> <a href="https://huggingface.co/nvidia/PixelDiT2-ImageNet"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-ImageNet-yellow" alt="ImageNet models"></a> |
| **PixelDiT: Pixel Diffusion Transformers for Image Generation** | **CVPR 2026 Oral**<br><a href="https://paperswithcode.co/conferences/cvpr-2026/best-paper-finalists"><img src="https://img.shields.io/badge/%F0%9F%8F%86_CVPR_2026-Best_Paper_Finalist-gold" alt="CVPR 2026 Best Paper Finalist"></a> | <a href="https://pixeldit.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_-Project-2ea44f" alt="Project"></a> <a href="https://arxiv.org/abs/2511.20645"><img src="https://img.shields.io/badge/%F0%9F%93%84_-arXiv-b31b1b.svg" alt="arXiv"></a> <a href="https://huggingface.co/nvidia/PixelDiT-ImageNet"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-ImageNet-yellow" alt="ImageNet models"></a> <a href="https://huggingface.co/nvidia/PixelDiT-1300M-1024px"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-T2I-yellow" alt="T2I model"></a><br> <a href="https://huggingface.co/Comfy-Org/PixelDiT"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FComfy-Org%2FPixelDiT%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Comfy-Org%2FPixelDiT%20downloads&color=yellow" alt="Comfy-Org/PixelDiT downloads"></a>  |

## Authors

**PixelDiT2**

[Yongsheng Yu](https://www.yongshengyu.com/) ·
[Wei Xiong](https://wxiong.me/) ·
[Yichen Sheng](https://shengcn.github.io/) ·
[Shiqiu Liu](http://behindthepixels.io/) ·
[Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

**PixelDiT**

[Yongsheng Yu](https://www.yongshengyu.com/) ·
[Wei Xiong](https://wxiong.me/) ·
[Weili Nie](https://weilinie.github.io/) ·
[Yichen Sheng](https://shengcn.github.io/) ·
[Shiqiu Liu](http://behindthepixels.io/) ·
[Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

**Affiliation**: NVIDIA · University of Rochester

**Project lead and main advisor**: [Wei Xiong](https://wxiong.me/)


## News

- **2026/09** — [PixelDiT2](https://pixeldit.github.io/pixeldit2/) is accepted to **NeurIPS 2026**, a representation-grounded pixel diffusion model that converges much faster.
- **2026/06** — [PixelDiT](https://pixeldit.github.io/) is a **CVPR 2026 Best Paper Finalist**.
- **2026/06** — Added [post-modulation for PiT blocks](c2i/README.md#training-stability-post-modulation-for-pit-blocks) to improve training stability.
- **2026/04** — PixelDiT training and inference code, and pretrained models, are released.
- **2026/02** — PixelDiT is accepted to **CVPR 2026 Oral**.
- **2025/11** — The [PixelDiT paper](https://arxiv.org/abs/2511.20645) is released.

<a id="pixeldit2"></a>
<a id="pixeldit"></a>
<a id="performance"></a>

## Introduction
**PixelDiT** is a series of work on pixel-space generative modeling, aiming to remove the need of VAEs which are widely used in traditional latent generative models and perform geneartion directly in pixel space. 
- **[PixelDiT](https://pixeldit.github.io/)**: Our first pixel-space diffusion models. We release both the models on ImageNet and a 1.3B Text-to-Image model that can generate 1024x1024 resolution images.
- **[PixelDiT2](https://pixeldit.github.io/pixeldit2/)**: We have a deeper investigation on how representation learning can guide the content generation for pixel diffusion. We propose to use representations to ground the generation process. It converges significantly faster and obtains better results on ImageNet. Please read the details in our [paper](https://arxiv.org/pdf/2609.24919). 

## Models and results

> Note: Our models are resumed every 4 hours, using the timestamp as the random seed each time. As a result, the final training outcome may have a slight gap compared to a continuous run without intermediate resumes.

### Class-to-image · ImageNet

| Model | Resolution | Epochs | FID ↓ | IS ↑ |
|---|:---:|---:|---:|---:|
| **PixelDiT2-H/16** | 256×256 | 600 | **1.46** | **301.6** |
| **PixelDiT2-H/16** | 512×512 | 680 | **1.48** | **295.7** |
| PixelDiT-XL | 256×256 | 320 | 1.61 | — |
| PixelDiT-XL | 512×512 | 850 | 1.81 | — |

Results use 50,000 samples. PixelDiT2 uses Heun with 50 steps; PixelDiT uses
FlowDPMSolver with 100 steps. The
[class-to-image guide](c2i/README.md) lists all checkpoints and the corresponding
sampling commands.

### Text-to-image · PixelDiT-T2I


| Resolution | GenEval ↑ | DPG-Bench ↑ |
|:---:|---:|---:|
| 512×512 | 0.78 | 83.7 |
| 1024×1024 | 0.74 | 83.5 |

See the [text-to-image guide](t2i/README.md) for training and inference.

**ComfyUI.** PixelDiT-T2I is also available in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The [Comfy-Org/PixelDiT](https://huggingface.co/Comfy-Org/PixelDiT) repository provides repackaged weights (bf16 and mxfp8) and a ready-to-use [text-to-image workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_pixeldit_t2i.json). The repository also hosts [PiD](https://github.com/nv-tlabs/PiD).


<a id="getting-started"></a>

## Quick start

Install the requirements for the model family you want to use, in separate
Python environments:

| Model family | Install from the repository root | Setup guide |
|---|---|---|
| PixelDiT2 | `python -m pip install -r requirements-pixeldit2.txt` | [PixelDiT2 environment](c2i/README.md#pixeldit2-environment) |
| PixelDiT (C2I / T2I) | `python -m pip install -r requirements-pixeldit1.txt` | [PixelDiT environment](c2i/README.md#pixeldit-environment) |

PixelDiT2 uses Python 3.10 and pinned PyTorch/CUDA dependencies. The PixelDiT
file retains the original dependency specification; `requirements.txt` remains
its compatibility entrypoint. C2I evaluation also differs: PixelDiT2 uses the
LTH14 torch-fidelity fork with JiT statistics, while PixelDiT uses ADM in a
separate evaluation environment. See the guides for details.

For class-to-image generation, both model families use `c2i/main.py`.
Choose a matching checkpoint and configuration in the
[model list](c2i/README.md#pretrained-models), then follow the shared
[inference](c2i/README.md#inference-and-evaluation) or
[training](c2i/README.md#training) workflow.

| Model family | Class-to-image configurations | Guide |
|---|---|---|
| PixelDiT2 | `pixeldit2_h16_in256.yaml`, `pixeldit2_h16_in512.yaml` | [Inference](c2i/README.md#pixeldit2-inference) |
| PixelDiT | `pix256_xl.yaml`, `pix512_xl.yaml` | [Inference](c2i/README.md#pixeldit-inference) |

### Use the model definitions in another project

The model code in `pixdit_core` can be installed with pip:

```bash
python -m pip install "git+https://github.com/NVlabs/PixelDiT"
# PixelDiT2's grounding encoder also needs timm:
python -m pip install "pixeldit[pixeldit2] @ git+https://github.com/NVlabs/PixelDiT"
```

Append `@<commit>` to the URL to pin a version. The package is named `pixeldit`
and is imported as `pixdit_core`:

```python
from pixdit_core.pixeldit_t2i import PixDiT_T2I    # PixelDiT text-to-image
from pixdit_core.pixeldit_c2i import PixDiT        # PixelDiT class-to-image
from pixdit_core.pixeldit2_c2i import PixelDiT2    # PixelDiT2 class-to-image
```

## Repository Structure

```
├── pixdit_core/      # Shared model definitions
│   ├── pixeldit2_c2i.py   # PixelDiT2  (single-path patch DiT + grounding)
│   ├── grounding.py       # PixelDiT2  frozen DINOv3 encoder and projection P_g
│   ├── pixeldit_c2i.py    # PixelDiT   (dual-level: patch DiT + pixel PiT)
│   └── pixeldit_t2i.py    # PixelDiT   text-to-image
├── tools/            # Checkpoint download, evaluation and GFLOPs computation
├── c2i/              # Class-to-image (PixelDiT and PixelDiT2)
└── t2i/              # Text-to-image
```

<details>
<summary>Compute model GFLOPs</summary>

### Compute GFLOPs

Measure single-forward-pass GFLOPs for any model in this repository (**run from
project root**). For PixelDiT2 the count includes the frozen grounding encoder.

```bash
# PixelDiT2 (ImageNet 256x256 and 512x512)
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in256.yaml
python tools/compute_flops.py --config c2i/configs/pixeldit2_h16_in512.yaml --height 512 --width 512
```

```bash
# PixelDiT C2I (ImageNet 256x256, default resolution)
python tools/compute_flops.py --config c2i/configs/pix256_xl.yaml
```

```bash
# PixelDiT T2I at 1024x1024
python tools/compute_flops.py --config t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml --height 1024 --width 1024
```

</details>

## Acknowledgements

We would like to thank the authors of [PixNerd](https://github.com/MCG-NJU/PixNerd) and [SANA](https://github.com/NVlabs/SANA) for sharing their code. We also thank the [SANA team](https://arxiv.org/pdf/2410.10629) for sharing their text-to-image training data.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026pixeldit2,
      title={PixelDiT2: Representation-Grounded Pixel Diffusion Transformers},
      author={Yongsheng Yu and Wei Xiong and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
      year={2026},
}

@inproceedings{yu2026pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
```
