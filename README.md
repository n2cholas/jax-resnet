# JAX ResNet - Implementations and Checkpoints for ResNet Variants

![Build & Tests](https://github.com/n2cholas/jax-resnet/workflows/Build%20and%20Tests/badge.svg)

A Flax (Linen) implementation of ResNet (He et al. 2015), Wide ResNet
(Zagoruyko & Komodakis 2016), ResNeXt (Xie et al. 2017), ResNet-D (He et al.
2020), and ResNeSt (Zhang et al. 2020). The code is modular so you can mix and
match the various stem, residual, and bottleneck implementations.

## Installation

You can install this package from PyPI:

```sh
pip install jax-resnet
```

Or directly from GitHub:

```sh
pip install --upgrade git+https://github.com/n2cholas/jax-resnet.git
```

## Usage

See the bottom of `jax-resnet/resnet.py` for the available aliases/options for
the ResNet variants (all models are in [Flax](https://github.com/google/flax))

Pretrained checkpoints from
[`torch.hub`](https://pytorch.org/docs/stable/hub.html) are available for the
following networks:

- ResNet [18, 34, 50, 101, 152]
- WideResNet [50, 101]
- ResNeXt [50, 101]
- ResNeSt [50-Fast, 50, 101, 200, 269]

The models are
[tested](https://github.com/n2cholas/jax-resnet/blob/main/tests/test_pretrained.py)
to have the same intermediate activations and outputs as the `torch.hub`
implementations, except ResNeSt-50 Fast, whose activations don't match exactly
but the final accuracy does.

A pretrained checkpoint for ResNetD-50 is available from
[fast.ai](https://github.com/fastai/fastai).
The activations do not match exactly, but the final accuracy matches.

```python
import jax.numpy as jnp
from jax_resnet import pretrained_resnest

ResNeSt50, variables = pretrained_resnest(50)
model = ResNeSt50()
out = model.apply(variables,
                  jnp.ones((32, 224, 224, 3)),  # ImageNet sized inputs.
                  mutable=False)  # Ensure `batch_stats` aren't updated.
```

You must install PyTorch yourself
([instructions](https://pytorch.org/get-started/locally/)) to use these
functions.

### Transfer Learning

To extract a subset of the model, you can use
`Sequential(model.layers[start:end])`.

The `slice_variables` function (found in in
[`common.py`](https://github.com/n2cholas/jax-resnet/blob/main/jax_resnet/common.py))
allows you to extract the corresponding subset of the variables dict. Check out
that docstring for more information.

## Checkpoint Accuracies

The top 1 and top 5 accuracies reported below are on the ImageNet2012
validation split.  The data was preprocessed as in the official [PyTorch
example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

|Model       | Size | Top 1 | Top 5 |
|------------|-----:|------:|------:|
|ResNet      |    18| 69.75%| 89.06%|
|            |    34| 73.29%| 91.42%|
|            |    50| 76.13%| 92.86%|
|            |   101| 77.37%| 93.53%|
|            |   152| 78.30%| 94.04%|
|Wide ResNet |    50| 78.48%| 94.08%|
|            |   101| 78.88%| 94.29%|
|ResNeXt     |    50| 77.60%| 93.70%|
|            |   101| 79.30%| 94.51%|
|ResNet-D    |    50| 77.57%| 93.85%|
<!--
|ResNeSt |    50| 80.97%| 95.38%|
|        |   101| 82.17%| 95.97%|
|        |   200| 82.35%| 96.11%|
|        |   269| 79.19%| 94.53%|
-->

The ResNeSt validation data was preprocessed as in
[zhang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt/blob/master/scripts/torch/verify.py).

|Model        | Size | Crop Size | Top 1 | Top 5 |
|-------------|-----:|----------:|------:|------:|
|ResNeSt-Fast |    50|        224| 80.53%| 95.34%|
|ResNeSt      |    50|        224| 81.05%| 95.42%|
|             |   101|        256| 82.82%| 96.32%|
|             |   200|        320| 83.84%| 96.86%|
|             |   269|        416| 84.53%| 96.98%|

## References

- [Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang,
  Shaoqing Ren, Jian Sun. _arXiv 2015_.](https://arxiv.org/abs/1512.03385)
- [Wide Residual Networks. Sergey Zagoruyko, Nikos Komodakis. _BMVC
  2016_](https://arxiv.org/abs/1605.07146)
- [Aggregated Residual Transformations for Deep Neural Networks. Saining Xie,
  Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. _CVPR
  2017_.](https://arxiv.org/abs/1611.05431)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks.
  Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. _CVPR
  2019_.](https://arxiv.org/abs/1812.01187)
- [ResNeSt: Split-Attention Networks. Hang Zhang, Chongruo Wu, Zhongyue Zhang,
  Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha,
  Mu Li, Alexander Smola. _arXiv 2020_.](https://arxiv.org/abs/2004.08955)
