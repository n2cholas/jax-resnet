from typing import Iterable, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

from .common import ConvBlock, ModuleDef


def rsoftmax(x, radix, cardinality):
    # (batch_size, features) -> (batch_size, features)
    batch = x.shape[0]
    if radix > 1:
        x = x.reshape((batch, cardinality, radix, -1)).swapaxes(1, 2)
        return nn.softmax(x, axis=1).reshape((batch, -1))
    else:
        return nn.sigmoid(x)


class SplAtConv2d(nn.Module):
    channels: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    groups: int = 1
    radix: int = 2
    reduction_factor: int = 4

    conv_block_cls: ModuleDef = ConvBlock
    cardinality: int = groups

    # Match extra bias here:
    # github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py#L39
    match_reference: bool = False

    @nn.compact
    def __call__(self, x):
        inter_channels = max(x.shape[-1] * self.radix // self.reduction_factor, 32)

        conv_block = self.conv_block_cls(self.channels * self.radix,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         groups=self.groups * self.radix,
                                         padding=self.padding)
        conv_cls = conv_block.conv_cls  # type: ignore
        x = conv_block(x)

        if self.radix > 1:
            # torch split takes split_size: int(rchannel//self.radix)
            # jnp split takes num sections: self.radix
            split = jnp.split(x, self.radix, axis=-1)
            gap = sum(split)
        else:
            gap = x

        gap = gap.mean((1, 2), keepdims=True)  # type: ignore # global average pool

        # Remove force_conv_bias after resolving
        # github.com/zhanghang1989/ResNeSt/issues/125
        gap = self.conv_block_cls(inter_channels,
                                  kernel_size=(1, 1),
                                  groups=self.cardinality,
                                  force_conv_bias=self.match_reference)(gap)

        attn = conv_cls(self.channels * self.radix,
                        kernel_size=(1, 1),
                        feature_group_count=self.cardinality)(gap)  # n x 1 x 1 x c
        attn = attn.reshape((x.shape[0], -1))
        attn = rsoftmax(attn, self.radix, self.cardinality)
        attn = attn.reshape((x.shape[0], 1, 1, -1))

        if self.radix > 1:
            attns = jnp.split(attn, self.radix, axis=-1)
            out = sum(a * s for a, s in zip(attns, split))
        else:
            out = attn * x

        return out
