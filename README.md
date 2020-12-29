# JAX ResNet - Implementation of ResNet, ResNet-D, and ResNeSt in Flax

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
  and SplAtConv2d.
- [x] Verified parameter count for ResNeSt and ResNeSt-Fast.

To-Do:

- [ ] Verify ResNeSt and ResNeSt-Fast intermediate shapes.
- [ ] Train all three models on ImageNet + release checkpoints
- [ ] ...probably lots more.
