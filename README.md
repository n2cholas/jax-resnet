# JAX ResNet - Implementation of ResNet, ResNet-D, and ResNeSt in Flax

![Build & Tests](https://github.com/n2cholas/jax-resnet/workflows/Build%20and%20Tests/badge.svg)

Work in progress!!

A Flax (Linen) implementation of ResNet (He, Kaiming, et al. 2015), ResNet-D
(He, Tong et al. 2020), and ResNest (Zhang, Hang et al. 2020). The code is
modular so you can mix and match the various stem, residual, and bottleneck
implementations.

## Installation

```sh
pip install --upgrade git+https://github.com/n2cholas/jax-resnet.git
```

## Usage

See the bottom of `jax-resnet/resnet.py` for the available aliases for the
ResNet variants (all models are built using
[Flax](https://github.com/google/flax))

For ResNeSt[50, 101, 200, 269], this library can load pretrained imagenet
weights from [`torch.hub`](https://pytorch.org/hub/pytorch_vision_resnest/).
The model is unit-tested to have the same intermediate activations as the
official [PyTorch implementation](https://github.com/zhanghang1989/ResNeSt). To
use this, ensure you have PyTorch installed, then:

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

## References

- [Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang,
  Shaoqing Ren, Jian Sun. _arXiv 2015_.](https://arxiv.org/abs/1512.03385)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks.
  Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. _CVPR
  2019_.](https://arxiv.org/abs/1812.01187)
- [ResNeSt: Split-Attention Networks. Hang Zhang, Chongruo Wu, Zhongyue Zhang,
  Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha,
  Mu Li, Alexander Smola. _arXiv 2020_.](https://arxiv.org/abs/2004.08955)
