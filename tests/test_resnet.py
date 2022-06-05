import jax
import pytest
from flax import linen as nn

from jax_resnet import *  # noqa

# github.com/google/flax/blob/master/linen_examples/imagenet/models.py
# matches https://pytorch.org/hub/pytorch_vision_resnet/
RESNET_PARAM_COUNTS = {
    18: 11689512,
    34: 21797672,
    50: 25557032,
    101: 44549160,
    152: 60192808,
}

# https://pytorch.org/hub/pytorch_vision_wide_resnet/
WIDE_RESNET_PARAM_COUNTS = {50: 68883240, 101: 126886696}

# https://pytorch.org/hub/pytorch_vision_resnext/
RESNEXT_PARAM_COUNTS = {50: 25028904, 101: 88791336}

# https://docs.fast.ai/vision.models.xresnet.html
RESNETD_PARAM_COUNTS = {
    18: 11708744,
    34: 21816904,
    50: 25576264,
    101: 44568392,
    152: 60212040,
}

# github.com/zhanghang1989/ResNeSt (torch version)
RESNEST_PARAM_COUNTS = {50: 27483240, 101: 48275016, 200: 70201544, 269: 110929480}


def n_params(model, init_shape=(1, 224, 224, 3)):
    init_array = jnp.ones(init_shape, jnp.float32)
    params = model.init(jax.random.PRNGKey(0), init_array)['params']
    return sum(map(jnp.size, jax.tree_leaves(params)))


@pytest.mark.parametrize('size', [18, 34, 50, 101, 152])
def _test_resnet_param_count(size):
    model = eval(f'ResNet{size}')(n_classes=1000)
    assert n_params(model) == RESNET_PARAM_COUNTS[size]


@pytest.mark.parametrize('size', [18, 50])
def test_resnet_param_count(size):
    _test_resnet_param_count(size)


@pytest.mark.slow
@pytest.mark.parametrize('size', [34, 101, 152])
def test_resnet_param_count_slow(size):
    _test_resnet_param_count(size)


@pytest.mark.parametrize('size', [50])
def test_wide_resnet_param_count(size):
    model = eval(f'WideResNet{size}')(n_classes=1000)
    assert n_params(model) == WIDE_RESNET_PARAM_COUNTS[size]


@pytest.mark.slow
@pytest.mark.parametrize('size', [101])
def test_wide_resnet_param_count_slow(size):
    model = eval(f'WideResNet{size}')(n_classes=1000)
    assert n_params(model) == WIDE_RESNET_PARAM_COUNTS[size]


@pytest.mark.parametrize('size', [50])
def test_resnext_param_count(size):
    model = eval(f'ResNeXt{size}')(n_classes=1000)
    assert n_params(model) == RESNEXT_PARAM_COUNTS[size]


@pytest.mark.slow
@pytest.mark.parametrize('size', [101])
def test_resnext_param_count_slow(size):
    model = eval(f'ResNeXt{size}')(n_classes=1000)
    assert n_params(model) == RESNEXT_PARAM_COUNTS[size]


def _test_resnetd_param_count(size):
    model = eval(f'ResNetD{size}')(n_classes=1000)
    assert n_params(model) == RESNETD_PARAM_COUNTS[size]


@pytest.mark.parametrize('size', [18, 50])
def test_resnetd_param_count(size):
    _test_resnetd_param_count(size)


@pytest.mark.slow
@pytest.mark.parametrize('size', [34, 101, 152])
def test_resnetd_param_count_slow(size):
    _test_resnetd_param_count(size)


@pytest.mark.parametrize('size', [50, 101, 200, 269])
def _test_resnest_param_count(size):
    block_cls = partial(ResNeStBottleneckBlock,
                        splat_cls=partial(SplAtConv2d, match_reference=True))
    model = eval(f'ResNeSt{size}')(n_classes=1000, block_cls=block_cls)
    assert n_params(model) == RESNEST_PARAM_COUNTS[size]


@pytest.mark.parametrize('size', [50])
def test_resnest_param_count(size):
    _test_resnest_param_count(size)


@pytest.mark.slow
@pytest.mark.parametrize('size', [101, 200, 269])
def test_resnest_param_count_slow(size):
    _test_resnest_param_count(size)


def test_resnest_fast_param_count():
    block_cls = partial(ResNeStBottleneckBlock,
                        splat_cls=partial(SplAtConv2d, match_reference=True))
    model = ResNeSt50Fast(n_classes=1000, block_cls=block_cls)
    # From github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/ablation.py#L48
    assert n_params(model) == 27483240


def test_splatconv2d_param_count():
    splat = SplAtConv2d(128, (3, 3), padding=[(1, 1), (1, 1)], match_reference=True)
    assert n_params(splat, init_shape=(1, 28, 28, 128)) == 172992


@pytest.mark.parametrize('start,end', [(0, 5), (0, None), (0, -3), (4, -2), (3, -1),
                                       (2, None)])
def test_slice_variables(start, end):
    model = ResNet18(n_classes=10)
    key = jax.random.PRNGKey(0)

    variables = model.init(key, jnp.ones((1, 224, 224, 3)))
    sliced_vars = slice_variables(variables, start, end)
    sliced_model = nn.Sequential(model.layers[start:end])

    # Need the correct number of input channels for slice:
    first = variables['params'][f'layers_{start}']['ConvBlock_0']['Conv_0']['kernel']
    slice_inp = jnp.ones((1, 224, 224, first.shape[2]))
    exp_sliced_vars = sliced_model.init(key, slice_inp)

    assert set(sliced_vars['params'].keys()) == set(exp_sliced_vars['params'].keys())
    assert set(sliced_vars['batch_stats'].keys()) == set(
        exp_sliced_vars['batch_stats'].keys())

    assert jax.tree_map(jnp.shape,
                        sliced_vars) == jax.tree_map(jnp.shape, exp_sliced_vars)

    sliced_model.apply(sliced_vars, slice_inp, mutable=False)
