# Baseline papers

These are the four objectives supported by the shared `CCUA-DDPM` runtime.
Fetch or refresh local reference PDFs with:

```bash
bash papers/fetch_papers.sh
```

PDFs are ignored by Git because they are third-party copyrighted works.

| Objective | Paper | Venue | Source |
|---|---|---|---|
| `ddpm` | Denoising Diffusion Probabilistic Models | NeurIPS 2020 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| `cbdm` | Class-Balancing Diffusion Models | CVPR 2023 | [arXiv:2305.00562](https://arxiv.org/abs/2305.00562) |
| `t2h` | Long-tailed Diffusion Models with Oriented Calibration | ICLR 2024 | [OpenReview NW2s5XXwXU](https://openreview.net/forum?id=NW2s5XXwXU) |
| `ccua` | Contrastive Conditional–Unconditional Alignment for Long-tailed Diffusion Model | arXiv | [arXiv:2507.09052](https://arxiv.org/abs/2507.09052) |

`ddpm` is the host with the long-tail-specific losses disabled. `cbdm`, `t2h`,
and `ccua` select their objective branches while retaining the same U-Net,
dataset, optimizer, checkpoint, DDIM sampler, and evaluator.

OpenReview blocks non-interactive PDF downloads for T2H. Supply a local copy
when needed:

```bash
LTX_PAPER_T2H_SRC=/path/to/t2h.pdf bash papers/fetch_papers.sh
```

Only `CCUA-DDPM/DDPM` is vendored. The CCUA paper's SiT/ImageNet-LT weights are
not transferred to this U-Net baseline; the U-Net defaults are used. Its
FLD/CLIP/DINOv2 metric stack is also not vendored: every row is scored by the
shared evaluator in this repository.
