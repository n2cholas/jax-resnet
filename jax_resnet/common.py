from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax.numpy as jnp

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
    def __call__(self, x):
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
            mutable = self.is_mutable_collection('batch_stats')
            x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


class Sequential(nn.Module):
    layers: Sequence[Union[nn.Module, Callable[[jnp.ndarray], jnp.ndarray]]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def slice_model(
    resnet: Sequential,
    start: int = 0,
    end: Optional[int] = None,
    *,
    variables: Optional[flax.core.FrozenDict] = None
) -> Union[Sequence, Tuple[Sequential, flax.core.FrozenDict]]:
    """Returns ResNet with a subset of the layers from indices [start, end).

    Args:
        resnet: A Sequential model (i.e. a flax.linen.Module with a `layers`
            attribute holding all the layers).
        start: integer indicating the first layer to keep.
        end: integer indicating the first layer to exclude (can be negative,
            has the same semantics as negative list indexing).
        variables: The flax.FrozenDict extract a subset of the layer state
            from.

    Returns:
        If variables is provided, a tuple with the sliced model and variables,
        otherwise just the sliced model.
    """
    if variables is None:
        return Sequential(resnet.layers[start:end])
    else:
        end_ind = end if end is not None else 0
        if end_ind < 0:
            end_ind = max(int(s.split('_')[-1]) for s in variables['params']) + end_ind

        sliced_variables: Dict[str, Any] = {}
        for k, var_dict in variables.items():  # usually params and batch_stats
            sliced_variables[k] = {}
            for i in range(start, end_ind):
                if f'layers_{i}' in var_dict:
                    sliced_variables[k][f'layers_{i}'] = var_dict[f'layers_{i}']

        return Sequential(resnet.layers[start:end]), flax.core.freeze(sliced_variables)
