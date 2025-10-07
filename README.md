# DeepSeek Sparse Attention PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org)

## What is this repo?

This repository provides **training & inference** of **DeepSeek Sparse Attention (DSA)** in **pure PyTorch**, with **zero-code** integration into popular models such as **Qwen**.

Official DeepSeek open-source release only offers:
* Inference kernels
* Support for DeepSeek architecture

We bridge the gap by offering:
* Full **training** pipeline
* **Plug-and-play** attention module for **Qwen** and other mainstream models

## Quick Start (3 lines)

```bash
git clone https://github.com/ZhengKai91/deepseek-sparse-attention-pytorch.git
cd deepseek-sparse-attention-pytorch
python demo.py          # run a minimal sparse-attention forward pass
```
demo.py provide a  minimal deepseek-sparse-attention generate content based on given inputs

## TODO List

- [ ] Training scripts (single-GPU & DDP)
- [ ] High-efficiency inference kernels (Triton / CUDA)
- [ ] Support for more models (LLaMA, ChatGLM, InternLM2...)

## Acknowledgments

Thanks to the DeepSeek Team for the V3.2 architecture innovations and original sparse-attention idea.

## Citation

If you find this repo useful, please cite:

```bibtex
@misc{deepseek-sparse-pytorch,
  title={PyTorch DeepSeek Sparse Attention (DSA) training & inference},
  author={Kai Zheng},
  year={2025},
  howpublished={\url{https://github.com/ZhengKai91/deepseek-sparse-attention-pytorch}}
}
```

## License

MIT License - see [LICENSE](https://github.com/ZhengKai91/deepseek-sparse-attention-pytorch/blob/main/LICENSE) for details.
