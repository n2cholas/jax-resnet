from functools import partial

import jax
import numpy as np
import pytest
import torch
# Below import makes JaxModuleTracker easier
from flax.linen import BatchNorm, Conv

from jax_resnet import *  # noqa


class JaxModuleTracker:
    def __init__(self):
        self.abs_sum, self.out_shape = {}, {}

    def __call__(tracker_self, cls):
        tracker_self.abs_sum[cls.__name__] = []
        tracker_self.out_shape[cls.__name__] = []
        # Need this to preserve the variables dict structure keys.
        exec(f'class {cls.__name__}(cls): pass')
        wrapped_cls = eval(cls.__name__)

        def call(self, x, *args, **kwargs):
            out = cls.__call__(self, x, *args, **kwargs)
            tracker_self.abs_sum[cls.__name__].append(np.abs(out).sum())
            tracker_self.out_shape[cls.__name__].append(out.shape)
            return out

        wrapped_cls.__call__ = call
        return wrapped_cls


class PTModuleTracker:
    def __init__(self):
        self.abs_sum, self.out_shape = {}, {}

    def __call__(self, layer, inp, out):
        name = layer.__class__.__name__
        if name not in self.abs_sum:
            self.abs_sum[name] = []
            self.out_shape[name] = []

        self.abs_sum[name].append(np.abs(out.detach().numpy()).sum())
        self.out_shape[name].append(tuple(out.shape))


@pytest.mark.parametrize('size', [50, 101, 200, 269])
def test_pretrained_resnest_outputs(size):
    jax_tracker = JaxModuleTracker()
    pt_tracker = PTModuleTracker()
    jax2pt_tracker = {  # Layer Name conversions
        'BatchNorm': 'BatchNorm2d',
        'ResNeStBottleneckBlock': 'Bottleneck',
        'SplAtConv2d': 'SplAtConv2d',
        'Conv': 'Conv2d',
        'ResNeStStem': 'ReLU',
    }

    # Create JAX Model and track all the modules
    block_cls = partial(jax_tracker(ResNeStBottleneckBlock),
                        splat_cls=partial(jax_tracker(SplAtConv2d),
                                          match_reference=True))
    conv_block_cls = partial(ConvBlock,
                             conv_cls=jax_tracker(Conv),
                             norm_cls=partial(jax_tracker(BatchNorm), momentum=0.9))
    stem_cls = stem_cls = partial(jax_tracker(ResNeStStem),
                                  stem_width=(32 if size == 50 else 64))
    jnet = eval(f'ResNeSt{size}')(n_classes=1000,
                                  block_cls=block_cls,
                                  stem_cls=stem_cls,
                                  conv_block_cls=conv_block_cls)
    _, variables = pretrained_resnest(size)

    # Load PT Model and register hooks to track intermediate values and shapes
    tnet = torch.hub.load('zhanghang1989/ResNeSt', f'resnest{size}',
                          pretrained=True).eval()

    for layer in [tnet.layer1, tnet.layer2, tnet.layer3, tnet.layer4]:
        for bottleneck in layer:
            bottleneck.conv2.register_forward_hook(pt_tracker)  # SplAt2d
            bottleneck.register_forward_hook(pt_tracker)  # Bottleneck

    tnet.conv1[0].register_forward_hook(pt_tracker)  # Stem Conv
    tnet.conv1[1].register_forward_hook(pt_tracker)  # Stem BatchNorm
    tnet.conv1[3].register_forward_hook(pt_tracker)  # Stem Conv
    tnet.conv1[4].register_forward_hook(pt_tracker)  # Stem BatchNorm
    tnet.conv1[6].register_forward_hook(pt_tracker)  # Stem Conv
    tnet.bn1.register_forward_hook(pt_tracker)  # Stem BatchNorm
    tnet.relu.register_forward_hook(pt_tracker)  # Stem Output

    jout = jnet.apply(variables, jnp.ones((1, 224, 224, 3)), mutable=False, train=False)
    with torch.no_grad():
        pout = tnet(torch.ones((1, 3, 224, 224))).detach().numpy()

    # Ensure outputs and shapes all match
    np.testing.assert_allclose(jout, pout, atol=0.01)
    atol = 1 if size == 50 else 3
    for jax_key, pt_key in jax2pt_tracker.items():
        assert (list(map(sorted, jax_tracker.out_shape[jax_key])) == list(
            map(sorted, pt_tracker.out_shape[pt_key])))
        np.testing.assert_allclose(jax_tracker.abs_sum[jax_key],
                                   pt_tracker.abs_sum[pt_key],
                                   atol=atol)


@pytest.mark.parametrize('size', [50, 101, 200, 269])
def test_pretrained_resnest_runs(size):
    model_cls, pretrained_variables = pretrained_resnest(size)
    model = model_cls()
    init_array = jnp.ones((1, 224, 224, 3), jnp.float32)
    init_variables = model.init(jax.random.PRNGKey(0), init_array)

    eq_tree = jax.tree_util.tree_multimap(lambda x, y: x.shape == y.shape,
                                          init_variables, pretrained_variables)
    assert jax.tree_util.tree_all(eq_tree)

    out = model.apply(pretrained_variables,
                      jnp.ones((1, 224, 224, 3)),
                      mutable=False,
                      train=False)
    assert out.shape == (1, 1000)
