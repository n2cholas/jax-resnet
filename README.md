# JAX ResNet - Implementation of ResNet, ResNet-D, and ResNeSt in Flax

![Build & Tests](https://github.com/n2cholas/jax-resnet/workflows/Build%20and%20Tests/badge.svg)

A Flax (Linen) implementation of ResNet (He, Kaiming, et al. 2015), ResNet-D
(He, Tong et al. 2020), and ResNest (Zhang, Hang et al. 2020). The code is
modular so you can mix and match the various stem, residual, and bottleneck
implementations.

## Installation

```sh
pip install --upgrade git+https://github.com/n2cholas/jax-resnet.git
```

## Usage

See the bottom of `jax-resnet/resnet.py` for the available aliases/options for
the ResNet variants (all models are in [Flax](https://github.com/google/flax))

Pretrained checkpoints from
[`torch.hub`](https://pytorch.org/docs/stable/hub.html) are available for the
following networks:

- ResNeSt[50, 101, 200, 269]
- ResNet[50, 101, 152]

The models are
[tested](https://github.com/n2cholas/jax-resnet/blob/main/tests/test_pretrained.py)
to have the same intermediate activations and outputs as the `torch.hub`
implementations.

```python
import jax.numpy as jnp
from jax_resnet import pretrained_resnest

ResNeSt50, variables = pretrained_resnest(50)
model = ResNeSt50()
out = model.apply(variables,
                  jnp.ones((32, 224, 224, 3)),  # ImageNet sized inputs.
                  mutable=False,  # Ensure `batch_stats` aren't updated.
                  train=False)  # Use running mean/var for batchnorm.
```

You must install PyTorch yourself
([instructions](https://pytorch.org/get-started/locally/)) to use those
functions.

A pretrained checkpoint for ResNetD-50 is available from
[fast.ai](https://github.com/fastai/fastai), however, the activations do not
match exactly. Feel free to use it via `pretrained_resnetd` (should be fine for
transfer learning). You must install fast.ai yourself
([instructions](https://docs.fast.ai/)) to use this function.

## References

- [Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang,
  Shaoqing Ren, Jian Sun. _arXiv 2015_.](https://arxiv.org/abs/1512.03385)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks.
  Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. _CVPR
  2019_.](https://arxiv.org/abs/1812.01187)
- [ResNeSt: Split-Attention Networks. Hang Zhang, Chongruo Wu, Zhongyue Zhang,
  Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha,
  Mu Li, Alexander Smola. _arXiv 2020_.](https://arxiv.org/abs/2004.08955)
