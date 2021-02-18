from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

from flax import linen as nn

ModuleDef = Callable[..., Callable]


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            x = self.norm_cls(use_running_average=not train, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x
