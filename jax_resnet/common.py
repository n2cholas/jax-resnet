from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import flax
import flax.linen as nn

ModuleDef = Callable[..., Callable]
# InitFn = Callable[[PRNGKey, Shape, DType], Array]
InitFn = Callable[[Any, Iterable[int], Any], Any]


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

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
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            mutable = self.is_mutable_collection('batch_stats')
            x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


def slice_variables(variables: Mapping[str, Any],
                    start: int = 0,
                    end: Optional[int] = None) -> flax.core.FrozenDict:
    """Returns variables dict correspond to a sliced model.

    You can retrieve the model corresponding to the slices variables via
    `Sequential(model.layers[start:end])`.

    The variables mapping should have the same structure as a Sequential
    model's variable dict (based on Flax):

        variables = {
            'group1': ['layers_a', 'layers_b', ...]
            'group2': ['layers_a', 'layers_b', ...]
            ...,
        }

    Typically, 'group1' and 'group2' would be 'params' and 'batch_stats', but
    they don't have to be. 'a, b, ...' correspond to the integer indices of the
    layers.

    Args:
        variables: A dict (typically a flax.core.FrozenDict) containing the
            model parameters and state.
        start: integer indicating the first layer to keep.
        end: integer indicating the first layer to exclude (can be negative,
            has the same semantics as negative list indexing).

    Returns:
        A flax.core.FrozenDict with the subset of parameters/state requested.
    """
    last_ind = max(int(s.split('_')[-1]) for s in variables['params'])
    if end is None:
        end = last_ind + 1
    elif end < 0:
        end += last_ind + 1

    sliced_variables: Dict[str, Any] = {}
    for k, var_dict in variables.items():  # usually params and batch_stats
        sliced_variables[k] = {
            f'layers_{i-start}': var_dict[f'layers_{i}']
            for i in range(start, end)
            if f'layers_{i}' in var_dict
        }

    return flax.core.freeze(sliced_variables)
