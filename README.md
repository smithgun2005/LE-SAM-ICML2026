# LE-SAM: Loss-Equated Sharpness-Aware Minimization

A PyTorch implementation of our paper Accepted by ICML2026 **Fix the Loss, Not the Radius: Rethinking the Adversarial Perturbation of Sharpness-Aware Minimization**(https://arxiv.org/abs/2605.10183)


### Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- tensorboard (optional, for logging)

## Quick Start

```bash
python lesam.py \
    --dataset cifar100 \
    --arch resnet18 \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.05 \
    --weight_decay 0.001 \
    --lesam_sigma 0.35 \
    --lesam_rho_max 999 \
```
## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wang2026fixlossradiusrethinking,
      title={Fix the Loss, Not the Radius: Rethinking the Adversarial Perturbation of Sharpness-Aware Minimization}, 
      author={Jinping Wang and Qinhan Liu and Zhiwu Xie and Zhiqiang Gao},
      year={2026},
      eprint={2605.10183},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.10183}, 
}
```
## Acknowledgements

This codebase builds on [ESAM](https://github.com/dydjw9/Efficient_SAM)  Dataset setup scripts are adapted from those repositories.

