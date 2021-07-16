from enum import Enum
from functools import partial

import jax
import numpy as np
import pytest
import torch
from fastai.vision.models.xresnet import xresnet50
# Below imports makes implementing JaxModuleTracker easier
from flax.linen import BatchNorm, Conv

from jax_resnet import *  # noqa


class JaxModuleTracker:
    def __init__(self):
        self.outputs = {}

    def __call__(tracker_self, cls):
        tracker_self.outputs[cls.__name__] = []
        # Need this to preserve the variables dict names.
        exec(f'class {cls.__name__}(cls): pass')
        wrapped_cls = eval(cls.__name__)

        def call(self, x, *args, **kwargs):
            out = cls.__call__(self, x, *args, **kwargs)
            tracker_self.outputs[cls.__name__].append(out)
            return out

        wrapped_cls.__call__ = call
        return wrapped_cls


class PTModuleTracker:
    def __init__(self):
        self.outputs = {}

    def __call__(self, layer, inp, out):
        name = layer.__class__.__name__
        if name not in self.outputs:
            self.outputs[name] = []

        self.outputs[name].append(out.detach().permute(0, 2, 3, 1).numpy())


class RNType(Enum):
    # Contains ResNets that use the vanilla ResNetBottleneckBlock class
    resnet = 1
    wide_resnet = 2
    resnext = 3


def _test_pretrained(size, pretrained_fn):
    model_cls, pretrained_vars = pretrained_fn(size)
    model = model_cls()
    arr = jnp.ones((1, 224, 224, 3), jnp.float32)
    init_vars = model.init(jax.random.PRNGKey(0), arr)

    eq_tree = jax.tree_multimap(lambda x, y: x.shape == y.shape, init_vars,
                                pretrained_vars)
    assert jax.tree_util.tree_all(eq_tree)

    out = model.apply(pretrained_vars, arr, mutable=False)
    assert out.shape == (1, 1000)


@pytest.mark.parametrize('size, pretrained_fn', [
    (18, pretrained_resnet),
    (50, pretrained_resnet),
    (50, pretrained_wide_resnet),
    (50, pretrained_resnext),
    (50, pretrained_resnetd),
    (50, pretrained_resnest),
    (50, partial(pretrained_resnest, fast=True)),
])
def test_pretrained(size, pretrained_fn):
    _test_pretrained(size, pretrained_fn)


@pytest.mark.slow
@pytest.mark.parametrize('size, pretrained_fn', [
    (34, pretrained_resnet),
    (101, pretrained_resnet),
    (152, pretrained_resnet),
    (101, pretrained_wide_resnet),
    (101, pretrained_resnext),
    (101, pretrained_resnest),
    (200, pretrained_resnest),
    (269, pretrained_resnest),
])
def test_pretrained_slow(size, pretrained_fn):
    _test_pretrained(size, pretrained_fn)


def _test_pretrained_resnet_activations(size, rntype):
    jtracker = JaxModuleTracker()
    ptracker = PTModuleTracker()

    jax2pt_names = {'ResNetStem': 'ReLU'}  # Layer Name conversions
    if size >= 50:
        jax2pt_names['ResNetBottleneckBlock'] = 'Bottleneck'
        block_cls = jtracker(ResNetBottleneckBlock)
    else:
        assert rntype == RNType.resnet
        jax2pt_names['ResNetBlock'] = 'BasicBlock'
        block_cls = jtracker(ResNetBlock)

    conv_block_cls = partial(ConvBlock,
                             conv_cls=jtracker(Conv),
                             norm_cls=partial(jtracker(BatchNorm), momentum=0.9))
    stem_cls = partial(jtracker(ResNetStem), conv_block_cls=conv_block_cls)
    kwargs = {'stem_cls': stem_cls, 'n_classes': 1000}

    if rntype == RNType.wide_resnet:
        jnet = eval(f'WideResNet{size}')(block_cls=partial(block_cls, expansion=2),
                                         **kwargs)
        _, variables = pretrained_wide_resnet(size)
        thub_name = f'wide_resnet{size}_2'
    elif rntype == RNType.resnext:
        block_cls = partial(block_cls, groups=32, base_width=(4 if size == 50 else 8))
        jnet = eval(f'ResNeXt{size}')(block_cls=block_cls, **kwargs)
        _, variables = pretrained_resnext(size)
        thub_name = 'resnext50_32x4d' if size == 50 else 'resnext101_32x8d'
    else:
        jnet = eval(f'ResNet{size}')(block_cls=block_cls, **kwargs)
        _, variables = pretrained_resnet(size)
        thub_name = f'resnet{size}'

    pnet = torch.hub.load('pytorch/vision:v0.10.0', thub_name, pretrained=True).eval()

    for layer in [pnet.layer1, pnet.layer2, pnet.layer3, pnet.layer4]:
        for block in layer:
            block.register_forward_hook(ptracker)  # Block
    pnet.relu.register_forward_hook(ptracker)  # Stem ReLU

    jout = jnet.apply(variables, jnp.ones((1, 224, 224, 3)), mutable=False)
    with torch.no_grad():
        pout = pnet(torch.ones((1, 3, 224, 224))).numpy()

    # Ensure outputs and shapes all match
    for jkey, pkey in jax2pt_names.items():
        for jact, pact in zip(jtracker.outputs[jkey], ptracker.outputs[pkey]):
            np.testing.assert_allclose(jact, pact, atol=0.001)
    np.testing.assert_allclose(jout, pout, atol=0.0001)


@pytest.mark.parametrize('size, rntype', [(18, RNType.resnet), (50, RNType.resnet),
                                          (50, RNType.wide_resnet),
                                          (50, RNType.resnext)])
def test_pretrained_resnet_activations(size, rntype):
    _test_pretrained_resnet_activations(size, rntype)


@pytest.mark.slow
@pytest.mark.parametrize('size, rntype', [(34, RNType.resnet), (101, RNType.resnet),
                                          (101, RNType.wide_resnet),
                                          (101, RNType.resnext), (152, RNType.resnet)])
def test_pretrained_resnet_activations_slow(size, rntype):
    _test_pretrained_resnet_activations(size, rntype)


@pytest.mark.parametrize('size', [50])
def test_pretrained_resnetd_activation_shapes(size):
    jax2pt_names = {
        'ResNetDStem': 'ConvLayer',
        'ResNetDSkipConnection': 'Sequential',
        'ResNetDBottleneckBlock': 'ResBlock',
    }
    jtracker = JaxModuleTracker()
    ptracker = PTModuleTracker()

    stem_cls = partial(jtracker(ResNetDStem))
    block_cls = partial(jtracker(ResNetDBottleneckBlock),
                        skip_cls=jtracker(ResNetDSkipConnection))
    jnet = eval(f'ResNetD{size}')(n_classes=1000,
                                  block_cls=block_cls,
                                  stem_cls=stem_cls)
    _, variables = pretrained_resnetd(size)

    pnet = xresnet50(pretrained=True).eval()
    for b, n_blocks in enumerate(STAGE_SIZES[size], 4):
        for i in range(n_blocks):
            pnet[b][i].register_forward_hook(ptracker)  # Bottleneck
            pnet[b][i].idpath.register_forward_hook(ptracker)  # Skip Connection

    pnet[2].register_forward_hook(ptracker)  # Stem

    jout = jnet.apply(variables, jnp.ones((1, 224, 224, 3)), mutable=False)
    with torch.no_grad():
        pout = pnet(torch.ones((1, 3, 224, 224))).numpy()

    # NOTE: Activation values currently do not match.
    for jkey, pkey in jax2pt_names.items():
        for jact, pact in zip(jtracker.outputs[jkey], ptracker.outputs[pkey]):
            # np.testing.assert_allclose(jact, pact, atol=0.001)
            assert jact.shape == pact.shape, f'{jkey}: {jact.shape}, {pact.shape}'
    assert jout.shape == pout.shape, f'output: {jout.shape}, {pout.shape}'
    # np.testing.assert_allclose(jout, pout, atol=0.0001)


def _test_pretrained_resnest_activations(size):
    jtracker = JaxModuleTracker()
    ptracker = PTModuleTracker()
    jax2pt_names = {  # Layer Name conversions
        'ResNeStBottleneckBlock': 'Bottleneck',
        'SplAtConv2d': 'SplAtConv2d',
        'Conv': 'Conv2d',
        'ResNetDStem': 'ReLU',
    }

    # Create JAX Model and track all the modules
    block_cls = partial(jtracker(ResNeStBottleneckBlock),
                        splat_cls=partial(jtracker(SplAtConv2d), match_reference=True))
    conv_block_cls = partial(ConvBlock, conv_cls=jtracker(Conv))
    stem_cls = partial(jtracker(ResNetDStem),
                       conv_block_cls=conv_block_cls,
                       stem_width=(32 if size == 50 else 64))
    jnet = eval(f'ResNeSt{size}')(n_classes=1000,
                                  block_cls=block_cls,
                                  stem_cls=stem_cls,
                                  conv_block_cls=conv_block_cls)
    _, variables = pretrained_resnest(size)

    # Load PT Model and register hooks to track intermediate values and shapes
    pnet = torch.hub.load('zhanghang1989/ResNeSt', f'resnest{size}',
                          pretrained=True).eval()

    for layer in [pnet.layer1, pnet.layer2, pnet.layer3, pnet.layer4]:
        for bottleneck in layer:
            bottleneck.conv2.register_forward_hook(ptracker)  # SplAt2d
            bottleneck.register_forward_hook(ptracker)  # Bottleneck

    pnet.conv1[0].register_forward_hook(ptracker)  # Stem Conv
    pnet.conv1[3].register_forward_hook(ptracker)  # Stem Conv
    pnet.conv1[6].register_forward_hook(ptracker)  # Stem Conv
    pnet.relu.register_forward_hook(ptracker)  # Stem Output

    jout = jnet.apply(variables, jnp.ones((1, 224, 224, 3)), mutable=False)
    with torch.no_grad():
        pout = pnet(torch.ones((1, 3, 224, 224))).numpy()

    # Ensure outputs and shapes all match
    for jkey, pkey in jax2pt_names.items():
        for jact, pact in zip(jtracker.outputs[jkey], ptracker.outputs[pkey]):
            np.testing.assert_allclose(jact, pact, atol=0.001)
    np.testing.assert_allclose(jout, pout, atol=0.0001)


@pytest.mark.parametrize('size', [50])
def test_pretrained_resnest_activations(size):
    # Fast variant does not match activations exactly.
    _test_pretrained_resnest_activations(size)


@pytest.mark.slow
@pytest.mark.parametrize('size', [101, 200, 269])
def test_pretrained_resnest_activations_slow(size):
    _test_pretrained_resnest_activations(size)
