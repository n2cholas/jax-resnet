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

## Progress

Done:

- [x] Verified intermediate shapes and parameter counts for ResNet, ResNet-D,
  SplAtConv2d, ResNeSt, and ResNeSt-Fast.

To-Do:

- [ ] Train all three models on ImageNet + release checkpoints
- [ ] ...probably lots more.

## References

- [Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang,
  Shaoqing Ren, Jian Sun. _arXiv 2015_.](https://arxiv.org/abs/1512.03385)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks.
  Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. _CVPR
  2019_.](https://arxiv.org/abs/1812.01187)
- [ResNeSt: Split-Attention Networks. Hang Zhang, Chongruo Wu, Zhongyue Zhang,
  Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha,
  Mu Li, Alexander Smola. _arXiv 2020_.](https://arxiv.org/abs/2004.08955)
