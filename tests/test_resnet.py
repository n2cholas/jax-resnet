import jax
import pytest

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

# https://docs.fast.ai/vision.models.xresnet.html
RESNETD_PARAM_COUNTS = {
    18: 11708744,
    34: 21816904,
    50: 25576264,
    101: 44568392,
    152: 60212040,
}

# github.com/zhanghang1989/ResNeSt (torch version)
# NOTE: might change after github.com/zhanghang1989/ResNeSt/issues/125
RESNEST_PARAM_COUNTS = {50: 27483240, 101: 48275016, 200: 70201544, 269: 110929480}


def n_params(model, init_shape=(1, 224, 224, 3)):
    init_array = jnp.ones(init_shape, jnp.float32)
    params = model.init(jax.random.PRNGKey(0), init_array)['params']
    return jax.tree_util.tree_reduce(lambda x, y: x + y,
                                     jax.tree_util.tree_map(lambda x: x.size, params))


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


def test_resnest_stem_param_count():
    assert n_params(ResNeStStem()) == 112832


@pytest.mark.parametrize('cls', [ResNet18, ResNeSt50Fast, ResNetD101])
def test_slice_model(cls):
    model = cls(n_classes=1000)
    init_array = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), init_array)
    model, variables = slice_model(model, 0, 5, variables=variables)
    layer_nums = [int(s.split('_')[-1]) for s in variables['params'].keys()]
    assert max(layer_nums) == 4
    assert min(layer_nums) == 0
    assert len(model.layers) == 5


@pytest.mark.parametrize('cls', [ResNet18, ResNeSt50Fast, ResNetD101])
def test_slice_model_no_variables(cls):
    model = cls(n_classes=1000)
    model = slice_model(model, 0, 5)
    assert len(model.layers) == 5


@pytest.mark.parametrize('cls', [ResNet18, ResNeSt50Fast, ResNetD101])
def test_slice_model_no_end(cls):
    model = cls(n_classes=1000)
    init_len = len(model.layers)
    model = slice_model(model, 5)
    assert len(model.layers) == init_len - 5


@pytest.mark.parametrize('cls', [ResNet18, ResNeSt50Fast, ResNetD101])
def test_slice_model_no_start(cls):
    model = cls(n_classes=1000)
    init_len = len(model.layers)
    model = slice_model(model, end=-3)
    assert len(model.layers) == init_len - 3
