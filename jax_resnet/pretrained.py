import warnings
from functools import partial
from typing import Dict, Mapping, Sequence, Tuple

from flax.core import freeze
from flax.traverse_util import unflatten_dict

from . import resnet
from .common import ModuleDef
from .splat import SplAtConv2d


def pretrained_resnet(size: int) -> Tuple[ModuleDef, Mapping]:
    """Returns returns variables for ResNest from torch.hub.

    Args:
        size: 50, 101 or 152.

    Returns:
        Module Class and variables dictionary for Flax ResNet.
    """
    try:
        import torch
    except ImportError:
        raise ImportError('Install `torch` to use this function. You may also need to '
                          '`pip install fvcore` due to this issue: '
                          'https://github.com/pytorch/pytorch/issues/53948')

    if size not in (50, 101, 152):
        raise ValueError('Ensure size is one of (50, 101, 200, 269)')

    pt2jax: Dict[str, Sequence[str]] = {}
    state_dict = torch.hub.load('pytorch/vision:v0.6.0',
                                f'resnet{size}',
                                pretrained=True).state_dict()

    add_bn = _get_add_bn(pt2jax)

    def bname(num):
        return f'layers_{num}'

    pt2jax['conv1.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))

    b_ind = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            for j in range(3):
                pt2jax[f'layer{b}.{i}.conv{j+1}.weight'] = ('params', bname(b_ind),
                                                            f'ConvBlock_{j}', 'Conv_0',
                                                            'kernel')
                add_bn(f'layer{b}.{i}.bn{j+1}',
                       (bname(b_ind), f'ConvBlock_{j}', 'BatchNorm_0'))

            if f'layer{b}.{i}.downsample.0.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.0.weight'] = ('params', bname(b_ind),
                                                               'ResNetSkipConnection_0',
                                                               'ConvBlock_0', 'Conv_0',
                                                               'kernel')
                add_bn(f'layer{b}.{i}.downsample.1',
                       (bname(b_ind), 'ResNetSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            b_ind += 1

    b_ind += 1
    pt2jax['fc.weight'] = ('params', bname(b_ind), 'kernel')
    pt2jax['fc.bias'] = ('params', bname(b_ind), 'bias')

    variables = _pytorch_to_jax_params(pt2jax, state_dict, ('fc.weight',))
    model_cls = partial(getattr(resnet, f'ResNet{size}'), n_classes=1000)
    return model_cls, freeze(unflatten_dict(variables))


def pretrained_resnetd(size: int) -> Tuple[ModuleDef, Mapping]:
    """Returns returns variables for ResNet from fastai.

    Args:
        size: 50.

    Returns:
        Module Class and variables dictionary for Flax ResNetD.
    """
    warnings.warn('This pretrained model\'s activations do not match the FastAI '
                  'reference implementation _exactly_. The model should still be '
                  'fine for transfer learning.')
    try:
        from fastai.vision.models.xresnet import xresnet50
    except ImportError:
        raise ImportError('Install `fastai>=2.2` to use this function.')

    if size != 50:
        raise ValueError('Ensure `size` is 50')

    try:
        state_dict = xresnet50(pretrained=True).state_dict()
    except Exception:
        raise RuntimeError('Failed to load pretrained PyTorch model. '
                           'Upgrade to `fastai>=2.2`')

    pt2jax: Dict[str, Sequence[str]] = {}
    add_bn = _get_add_bn(pt2jax)

    def add_convblock(pt_layer, jax_layer):
        pt2jax[f'{pt_layer}.0.weight'] = ('params', *jax_layer, 'Conv_0', 'kernel')
        add_bn(f'{pt_layer}.1', (*jax_layer, 'BatchNorm_0'))

    def bname(num):
        return f'layers_{num}'

    for i in range(3):
        add_convblock(i, ('layers_0', f'ConvBlock_{i}'))

    b_ind = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 4):
        for i in range(n_blocks):
            for j in range(3):
                add_convblock(f'{b}.{i}.convpath.{j}', (bname(b_ind), f'ConvBlock_{j}'))

            if f'{b}.{i}.idpath.0.0.weight' in state_dict:  # no average pool
                add_convblock(f'{b}.{i}.idpath.0',
                              (bname(b_ind), 'ResNetDSkipConnection_0', 'ConvBlock_0'))
            elif f'{b}.{i}.idpath.1.0.weight' in state_dict:  # with average pool
                add_convblock(f'{b}.{i}.idpath.1',
                              (bname(b_ind), 'ResNetDSkipConnection_0', 'ConvBlock_0'))

            b_ind += 1

    b_ind += 1
    pt2jax['11.weight'] = ('params', bname(b_ind), 'kernel')
    pt2jax['11.bias'] = ('params', bname(b_ind), 'bias')

    variables = _pytorch_to_jax_params(pt2jax, state_dict, ('11.weight',))
    model_cls = partial(getattr(resnet, f'ResNetD{size}'), n_classes=1000)
    return model_cls, freeze(unflatten_dict(variables))


def pretrained_resnest(size: int) -> Tuple[ModuleDef, Mapping]:
    """Returns returns variables for ResNest from torch.hub.

    Args:
        size: 50, 101, 200, or 269.

    Returns:
        Module Class and variables dictionary for Flax ResNeSt.
    """
    try:
        import torch
    except ImportError:
        raise ImportError('Install `torch` to use this function.')

    if size not in (50, 101, 200, 269):
        raise ValueError('Ensure size is one of (50, 101, 200, 269)')

    pt2jax: Dict[str, Sequence[str]] = {}
    state_dict = torch.hub.load('zhanghang1989/ResNeSt',
                                f'resnest{size}',
                                pretrained=True).state_dict()

    add_bn = _get_add_bn(pt2jax)

    def bname(num):
        return f'layers_{num}'

    # Stem
    pt2jax['conv1.0.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('conv1.1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))
    pt2jax['conv1.3.weight'] = ('params', 'layers_0', 'ConvBlock_1', 'Conv_0', 'kernel')
    add_bn('conv1.4', ('layers_0', 'ConvBlock_1', 'BatchNorm_0'))
    pt2jax['conv1.6.weight'] = ('params', 'layers_0', 'ConvBlock_2', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_2', 'BatchNorm_0'))

    b_ind = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            pt2jax[f'layer{b}.{i}.conv1.weight'] = ('params', bname(b_ind),
                                                    'ConvBlock_0', 'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.bn1', (bname(b_ind), 'ConvBlock_0', 'BatchNorm_0'))

            # splat
            pt2jax[f'layer{b}.{i}.conv2.conv.weight'] = ('params', bname(b_ind),
                                                         'SplAtConv2d_0', 'ConvBlock_0',
                                                         'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.conv2.bn0',
                   (bname(b_ind), 'SplAtConv2d_0', 'ConvBlock_0', 'BatchNorm_0'))
            pt2jax[f'layer{b}.{i}.conv2.fc1.weight'] = ('params', bname(b_ind),
                                                        'SplAtConv2d_0', 'ConvBlock_1',
                                                        'Conv_0', 'kernel')
            pt2jax[f'layer{b}.{i}.conv2.fc1.bias'] = ('params', bname(b_ind),
                                                      'SplAtConv2d_0', 'ConvBlock_1',
                                                      'Conv_0', 'bias')
            add_bn(f'layer{b}.{i}.conv2.bn1',
                   (bname(b_ind), 'SplAtConv2d_0', 'ConvBlock_1', 'BatchNorm_0'))
            pt2jax[f'layer{b}.{i}.conv2.fc2.weight'] = ('params', bname(b_ind),
                                                        'SplAtConv2d_0', 'Conv_0',
                                                        'kernel')
            pt2jax[f'layer{b}.{i}.conv2.fc2.bias'] = ('params', bname(b_ind),
                                                      'SplAtConv2d_0', 'Conv_0', 'bias')

            # rest
            pt2jax[f'layer{b}.{i}.conv3.weight'] = ('params', bname(b_ind),
                                                    'ConvBlock_1', 'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.bn3', (bname(b_ind), 'ConvBlock_1', 'BatchNorm_0'))

            # Downsample
            if f'layer{b}.{i}.downsample.1.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.1.weight'] = (
                    'params', bname(b_ind), 'ResNeStSkipConnection_0', 'ConvBlock_0',
                    'Conv_0', 'kernel')
                add_bn(f'layer{b}.{i}.downsample.2',
                       (bname(b_ind), 'ResNeStSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            b_ind += 1

    b_ind += 1
    pt2jax['fc.weight'] = ('params', bname(b_ind), 'kernel')
    pt2jax['fc.bias'] = ('params', bname(b_ind), 'bias')

    variables = _pytorch_to_jax_params(pt2jax, state_dict, ('fc.weight',))

    block_cls = partial(resnet.ResNeStBottleneckBlock,
                        splat_cls=partial(SplAtConv2d, match_reference=True))
    model_cls = partial(getattr(resnet, f'ResNeSt{size}'),
                        n_classes=1000,
                        block_cls=block_cls)
    return model_cls, freeze(unflatten_dict(variables))


def _pytorch_to_jax_params(pt2jax, state_dict, fc_keys):
    variables = {}
    for pt_name, jax_key in pt2jax.items():
        w = state_dict[pt_name].numpy()
        if w.ndim == 4:
            w = w.transpose((2, 3, 1, 0))
        elif pt_name in fc_keys:
            w = w.transpose()
        variables[jax_key] = w

    return variables


def _get_add_bn(pt2jax):
    def add_bn(pname, jprefix):
        pt2jax[f'{pname}.weight'] = ('params', *jprefix, 'scale')
        pt2jax[f'{pname}.bias'] = ('params', *jprefix, 'bias')
        pt2jax[f'{pname}.running_mean'] = ('batch_stats', *jprefix, 'mean')
        pt2jax[f'{pname}.running_var'] = ('batch_stats', *jprefix, 'var')

    return add_bn
