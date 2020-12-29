from functools import partial
from typing import Callable, Sequence, Tuple

from flax import linen as nn

from .common import ConvBlock, ModuleDef
from .splat import SplAtConv2d


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

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.conv_block_cls((x.shape[-1] + 1) * 8, strides=(2, 2))(x, train=train)
        x = self.conv_block_cls(64, strides=(1, 1))(x, train=train)
        x = self.conv_block_cls(64, strides=(1, 1))(x, train=train)
        return x


class ResNeStStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.conv_block_cls(self.stem_width, kernel_size=(3, 3),
                                strides=(2, 2))(x, train=train)
        x = self.conv_block_cls(self.stem_width, kernel_size=(3, 3),
                                strides=(1, 1))(x, train=train)
        x = self.conv_block_cls(self.stem_width * 2, kernel_size=(3, 3),
                                strides=(1, 1))(x, train=train)
        return x


class ResNetBlock(nn.Module):
    bottleneck: bool
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock

    def residual(self, x, main_shape, train: bool = True):
        if x.shape != main_shape:
            x = self.conv_block_cls(main_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x, train=train)
        return x

    @nn.compact
    def __call__(self, x, train: bool = True):
        if self.bottleneck:
            n_filters = self.n_hidden * 4
            y = self.conv_block_cls(self.n_hidden, kernel_size=(1, 1))(x, train=train)
            y = self.conv_block_cls(self.n_hidden, strides=self.strides)(y, train=train)
            y = self.conv_block_cls(n_filters, kernel_size=(1, 1),
                                    is_last=True)(y, train=train)
        else:
            n_filters = self.n_hidden
            y = self.conv_block_cls(self.n_hidden, strides=self.strides)(x, train=train)
            y = self.conv_block_cls(n_filters, is_last=True)(y, train=train)

        return self.activation(y + self.residual(x, y.shape, train=train))


class ResNetDBlock(ResNetBlock):
    def residual(self, x, main_shape, train: bool = True):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding='SAME')
        if x.shape[-1] != main_shape[-1]:
            x = self.conv_block_cls(main_shape[-1], (1, 1),
                                    activation=lambda y: y)(x, train=train)
        return x


class ResNeStBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    avg_pool_first: bool = False

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    splat_cls: ModuleDef = SplAtConv2d

    groups: int = 1  # cardinality
    radix: int = 2
    bottleneck_width: int = 64

    def setup(self):
        # TODO: implement groups != 1 and radix != 2
        assert self.groups == 1
        assert self.radix == 2

    def residual(self, x, main_shape, train: bool = True):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding='SAME')
        if x.shape[-1] != main_shape[-1]:
            x = self.conv_block_cls(main_shape[-1], (1, 1),
                                    activation=lambda y: y)(x, train=train)
        return x

    @nn.compact
    def __call__(self, x, train: bool = True):
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

        return self.activation(y + self.residual(x, y.shape, train=train))


class ResNet(nn.Module):
    block_cls: ModuleDef
    stage_sizes: Sequence[int]
    n_classes: int

    conv_block_cls: ModuleDef = ConvBlock
    stem_cls: ModuleDef = ResNetStem

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.stem_cls(self.conv_block_cls)(x, train=train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        filters_list = [64, 128, 256, 512]
        assert len(filters_list) == len(self.stage_sizes)
        for i, (n_filters, n_blocks) in enumerate(zip(filters_list, self.stage_sizes)):
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                x = self.block_cls(n_hidden=n_filters, strides=strides)(x, train=train)

        x = x.mean((-2, -3))  # global average pool
        return nn.Dense(self.n_classes)(x)


_r18_stages = [2, 2, 2, 2]
_r34_stages = [3, 4, 6, 3]
_r50_stages = [3, 4, 6, 3]
_r101_stages = [3, 4, 23, 3]
_r152_stages = [3, 8, 36, 3]
_r200_stages = [3, 24, 36, 3]
_r269_stages = [3, 30, 48, 8]

ResNetD18 = partial(ResNet,
                    stage_sizes=_r18_stages,
                    stem_cls=ResNetDStem,
                    block_cls=partial(ResNetDBlock, bottleneck=False))
ResNetD34 = partial(ResNet,
                    stage_sizes=_r34_stages,
                    stem_cls=ResNetDStem,
                    block_cls=partial(ResNetDBlock, bottleneck=False))
ResNetD50 = partial(ResNet,
                    stage_sizes=_r50_stages,
                    stem_cls=ResNetDStem,
                    block_cls=partial(ResNetDBlock, bottleneck=True))
ResNetD101 = partial(ResNet,
                     stage_sizes=_r101_stages,
                     stem_cls=ResNetDStem,
                     block_cls=partial(ResNetDBlock, bottleneck=True))
ResNetD152 = partial(ResNet,
                     stage_sizes=_r152_stages,
                     stem_cls=ResNetDStem,
                     block_cls=partial(ResNetDBlock, bottleneck=True))
ResNetD200 = partial(ResNet,
                     stage_sizes=_r200_stages,
                     stem_cls=ResNetDStem,
                     block_cls=partial(ResNetDBlock, bottleneck=True))

ResNet18 = partial(ResNet,
                   stage_sizes=_r18_stages,
                   stem_cls=ResNetStem,
                   block_cls=partial(ResNetBlock, bottleneck=False))
ResNet34 = partial(ResNet,
                   stage_sizes=_r34_stages,
                   stem_cls=ResNetStem,
                   block_cls=partial(ResNetBlock, bottleneck=False))
ResNet50 = partial(ResNet,
                   stage_sizes=_r50_stages,
                   stem_cls=ResNetStem,
                   block_cls=partial(ResNetBlock, bottleneck=True))
ResNet101 = partial(ResNet,
                    stage_sizes=_r101_stages,
                    stem_cls=ResNetStem,
                    block_cls=partial(ResNetBlock, bottleneck=True))
ResNet152 = partial(ResNet,
                    stage_sizes=_r152_stages,
                    stem_cls=ResNetStem,
                    block_cls=partial(ResNetBlock, bottleneck=True))
ResNet200 = partial(ResNet,
                    stage_sizes=_r200_stages,
                    stem_cls=ResNetStem,
                    block_cls=partial(ResNetBlock, bottleneck=True))

ResNeSt50Fast = partial(ResNet,
                        stage_sizes=_r50_stages,
                        stem_cls=partial(ResNeStStem, stem_width=32),
                        block_cls=partial(ResNeStBlock, avg_pool_first=True))
ResNeSt50 = partial(ResNet,
                    stage_sizes=_r50_stages,
                    stem_cls=partial(ResNeStStem, stem_width=32),
                    block_cls=ResNeStBlock)
ResNeSt101 = partial(ResNet,
                     stage_sizes=_r101_stages,
                     stem_cls=ResNeStStem,
                     block_cls=ResNeStBlock)
ResNeSt152 = partial(ResNet,
                     stage_sizes=_r152_stages,
                     stem_cls=ResNeStStem,
                     block_cls=ResNeStBlock)
ResNeSt200 = partial(ResNet,
                     stage_sizes=_r200_stages,
                     stem_cls=ResNeStStem,
                     block_cls=ResNeStBlock)
ResNeSt269 = partial(ResNet,
                     stage_sizes=_r269_stages,
                     stem_cls=ResNeStStem,
                     block_cls=ResNeStBlock)
