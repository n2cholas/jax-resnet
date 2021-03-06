from functools import partial
from typing import Callable, Optional, Sequence, Tuple

from flax import linen as nn

from .common import ConvBlock, ModuleDef
from .splat import SplAtConv2d

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, train: bool = True):
        return self.conv_block_cls(64,
                                   kernel_size=(7, 7),
                                   strides=(2, 2),
                                   padding=[(3, 3), (3, 3)])(x, train=train)


class ResNetDStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 32

    # If True, n_filters for first conv is (input_channels + 1) * 8
    adaptive_first_width: bool = True  # TODO: Find better name.

    @nn.compact
    def __call__(self, x, train: bool = True):
        cls = partial(self.conv_block_cls, kernel_size=(3, 3), padding=((1, 1), (1, 1)))
        first_width = (8 * (x.shape[-1] + 1)
                       if self.adaptive_first_width else self.stem_width)
        x = cls(first_width, strides=(2, 2))(x, train=train)
        x = cls(self.stem_width, strides=(1, 1))(x, train=train)
        x = cls(self.stem_width * 2, strides=(1, 1))(x, train=train)
        return x


class ResNeStStem(ResNetDStem):
    stem_width: int = 64  # TODO: make this the same as ResNetD?
    adaptive_first_width: bool = False


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape, train: bool = True):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x, train=train)
        return x


class ResNetDSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape, train: bool = True):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        if x.shape[-1] != out_shape[-1]:
            x = self.conv_block_cls(out_shape[-1], (1, 1),
                                    activation=lambda y: y)(x, train=train)
        return x


class ResNeStSkipConnection(ResNetDSkipConnection):
    # Inheritance to ensures our variables dict has the right names.
    pass


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x, train: bool = True):
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x, train=train)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y, train=train)
        return self.activation(y + self.skip_cls(self.strides)(x, y.shape, train=train))


class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x, train: bool = True):
        y = self.conv_block_cls(self.n_hidden, kernel_size=(1, 1))(x, train=train)
        y = self.conv_block_cls(self.n_hidden,
                                strides=self.strides,
                                padding=((1, 1), (1, 1)))(y, train=train)
        y = self.conv_block_cls(self.n_hidden * 4, kernel_size=(1, 1),
                                is_last=True)(y, train=train)
        return self.activation(y + self.skip_cls(self.strides)(x, y.shape, train=train))


class ResNetDBlock(ResNetBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNetDBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNeStBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNeStSkipConnection
    avg_pool_first: bool = False
    groups: int = 1  # cardinality
    radix: int = 2
    bottleneck_width: int = 64

    splat_cls: ModuleDef = SplAtConv2d

    @nn.compact
    def __call__(self, x, train: bool = True):
        # TODO: implement groups != 1 and radix != 2
        assert self.groups == 1
        assert self.radix == 2

        n_filters = self.n_hidden * 4
        group_width = int(self.n_hidden * (self.bottleneck_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x, train=train)

        if self.strides != (1, 1) and self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.splat_cls(group_width,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding=[(1, 1), (1, 1)],
                           groups=self.groups,
                           radix=self.radix)(y, train=train)

        if self.strides != (1, 1) and not self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.conv_block_cls(n_filters, kernel_size=(1, 1),
                                is_last=True)(y, train=train)

        return self.activation(y + self.skip_cls(self.strides)(x, y.shape, train=train))


class ResNet(nn.Module):
    block_cls: ModuleDef
    stage_sizes: Sequence[int]
    n_classes: int

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    conv_block_cls: ModuleDef = ConvBlock
    stem_cls: ModuleDef = ResNetStem
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1)))

    # When True, the model will propogate the top-level conv_cls and norm_cls
    # through the conv_block_cls to all the submodules (stem, bottleneck, etc).
    consistent_conv_block: bool = False
    backbone_only: bool = False  # When True, no GlobalAveragePool or Dense

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv_block_cls = partial(self.conv_block_cls,
                                 conv_cls=self.conv_cls,
                                 norm_cls=self.norm_cls)
        stem_cls, block_cls = self.stem_cls, self.block_cls
        if self.consistent_conv_block:
            stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
            # TODO: set conv_block_cls for skip_cls
            block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

        x = stem_cls()(x, train=train)
        x = self.pool_fn(x)

        for i, n_blocks in enumerate(self.stage_sizes):
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                x = block_cls(n_hidden=2**(i + 6), strides=strides)(x, train=train)

        if self.backbone_only:
            return x

        x = x.mean((-2, -3))  # global average pool
        return nn.Dense(self.n_classes)(x)


# yapf: disable
ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                   stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)

ResNetD18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)

ResNeSt50Fast = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                        stem_cls=partial(ResNeStStem, stem_width=32),
                        block_cls=partial(ResNeStBottleneckBlock, avg_pool_first=True))
ResNeSt50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=partial(ResNeStStem, stem_width=32),
                    block_cls=ResNeStBottleneckBlock)
ResNeSt101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=ResNeStStem, block_cls=ResNeStBottleneckBlock)
ResNeSt152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=ResNeStStem, block_cls=ResNeStBottleneckBlock)
ResNeSt200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=ResNeStStem, block_cls=ResNeStBottleneckBlock)
ResNeSt269 = partial(ResNet, stage_sizes=STAGE_SIZES[269],
                     stem_cls=ResNeStStem, block_cls=ResNeStBottleneckBlock)
