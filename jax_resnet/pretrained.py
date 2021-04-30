import warnings
from functools import partial
from typing import Dict, Mapping, Sequence, Tuple

from flax.core import freeze
from flax.traverse_util import unflatten_dict

from . import resnet
from .common import ModuleDef
from .splat import SplAtConv2d

# ResNet-D 50 from FastAI:
# github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py#L22
_RESNETD_URL = {50: 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'}

# ResNest from zhanghang1989/ResNeSt
# github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py#L16-L32
_RESNEST_URL_FMT = 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest{}-{}.pth'
_RESNEST_SHA256 = {50: '528c19ca', 101: '22405ba7', 200: '75117900', 269: '0cc87c48'}
_RESNEST_URL = {
    size: _RESNEST_URL_FMT.format(size, _RESNEST_SHA256[size][:8])
    for size in _RESNEST_SHA256.keys()
}


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
        raise ImportError('Install `torch` to use this function.')

    if size not in (50, 101, 152):
        raise ValueError('Ensure size is one of (50, 101, 200, 269)')

    pt2jax: Dict[str, Sequence[str]] = {}
    state_dict = torch.hub.load('pytorch/vision:v0.6.0',
                                f'resnet{size}',
                                pretrained=True).state_dict()

    add_bn = _get_add_bn(pt2jax)

    pt2jax['conv1.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))

    lyr = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            for j in range(3):
                pt2jax[f'layer{b}.{i}.conv{j+1}.weight'] = ('params', f'layers_{lyr}',
                                                            f'ConvBlock_{j}', 'Conv_0',
                                                            'kernel')
                add_bn(f'layer{b}.{i}.bn{j+1}',
                       (f'layers_{lyr}', f'ConvBlock_{j}', 'BatchNorm_0'))

            if f'layer{b}.{i}.downsample.0.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.0.weight'] = ('params',
                                                               f'layers_{lyr}',
                                                               'ResNetSkipConnection_0',
                                                               'ConvBlock_0', 'Conv_0',
                                                               'kernel')
                add_bn(f'layer{b}.{i}.downsample.1',
                       (f'layers_{lyr}', 'ResNetSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            lyr += 1

    lyr += 1
    pt2jax['fc.weight'] = ('params', f'layers_{lyr}', 'kernel')
    pt2jax['fc.bias'] = ('params', f'layers_{lyr}', 'bias')

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
        import torch
    except ImportError:
        raise ImportError('Install `torch` to use this function.')

    if size not in _RESNETD_URL.keys():
        raise ValueError(f'Ensure `size` is one of {tuple(_RESNETD_URL.keys())}')

    state_dict = torch.hub.load_state_dict_from_url(_RESNETD_URL[size],
                                                    map_location='cpu')['model']
    pt2jax: Dict[str, Sequence[str]] = {}
    add_bn = _get_add_bn(pt2jax)

    def add_convblock(pt_layer, jax_layer):
        pt2jax[f'{pt_layer}.0.weight'] = ('params', *jax_layer, 'Conv_0', 'kernel')
        add_bn(f'{pt_layer}.1', (*jax_layer, 'BatchNorm_0'))

    for i in range(3):
        add_convblock(i, ('layers_0', f'ConvBlock_{i}'))

    lyr = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 4):
        for i in range(n_blocks):
            for j in range(3):
                add_convblock(f'{b}.{i}.convs.{j}', (f'layers_{lyr}', f'ConvBlock_{j}'))

            if f'{b}.{i}.idconv.0.weight' in state_dict:
                add_convblock(
                    f'{b}.{i}.idconv',
                    (f'layers_{lyr}', 'ResNetDSkipConnection_0', 'ConvBlock_0'))

            lyr += 1

    lyr += 1
    pt2jax['10.weight'] = ('params', f'layers_{lyr}', 'kernel')
    pt2jax['10.bias'] = ('params', f'layers_{lyr}', 'bias')

    variables = _pytorch_to_jax_params(pt2jax, state_dict, ('10.weight',))
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

    if size not in _RESNEST_URL.keys():
        raise ValueError(f'Ensure `size` is one of {tuple(_RESNEST_URL.keys())}')

    state_dict = torch.hub.load_state_dict_from_url(_RESNEST_URL[size],
                                                    map_location='cpu')
    pt2jax: Dict[str, Sequence[str]] = {}
    add_bn = _get_add_bn(pt2jax)

    # Stem
    pt2jax['conv1.0.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('conv1.1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))
    pt2jax['conv1.3.weight'] = ('params', 'layers_0', 'ConvBlock_1', 'Conv_0', 'kernel')
    add_bn('conv1.4', ('layers_0', 'ConvBlock_1', 'BatchNorm_0'))
    pt2jax['conv1.6.weight'] = ('params', 'layers_0', 'ConvBlock_2', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_2', 'BatchNorm_0'))

    lyr = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            pt2jax[f'layer{b}.{i}.conv1.weight'] = ('params', f'layers_{lyr}',
                                                    'ConvBlock_0', 'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.bn1', (f'layers_{lyr}', 'ConvBlock_0', 'BatchNorm_0'))

            # splat
            pt2jax[f'layer{b}.{i}.conv2.conv.weight'] = ('params', f'layers_{lyr}',
                                                         'SplAtConv2d_0', 'ConvBlock_0',
                                                         'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.conv2.bn0',
                   (f'layers_{lyr}', 'SplAtConv2d_0', 'ConvBlock_0', 'BatchNorm_0'))
            pt2jax[f'layer{b}.{i}.conv2.fc1.weight'] = ('params', f'layers_{lyr}',
                                                        'SplAtConv2d_0', 'ConvBlock_1',
                                                        'Conv_0', 'kernel')
            pt2jax[f'layer{b}.{i}.conv2.fc1.bias'] = ('params', f'layers_{lyr}',
                                                      'SplAtConv2d_0', 'ConvBlock_1',
                                                      'Conv_0', 'bias')
            add_bn(f'layer{b}.{i}.conv2.bn1',
                   (f'layers_{lyr}', 'SplAtConv2d_0', 'ConvBlock_1', 'BatchNorm_0'))
            pt2jax[f'layer{b}.{i}.conv2.fc2.weight'] = ('params', f'layers_{lyr}',
                                                        'SplAtConv2d_0', 'Conv_0',
                                                        'kernel')
            pt2jax[f'layer{b}.{i}.conv2.fc2.bias'] = ('params', f'layers_{lyr}',
                                                      'SplAtConv2d_0', 'Conv_0', 'bias')

            # rest
            pt2jax[f'layer{b}.{i}.conv3.weight'] = ('params', f'layers_{lyr}',
                                                    'ConvBlock_1', 'Conv_0', 'kernel')
            add_bn(f'layer{b}.{i}.bn3', (f'layers_{lyr}', 'ConvBlock_1', 'BatchNorm_0'))

            # Downsample
            if f'layer{b}.{i}.downsample.1.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.1.weight'] = (
                    'params', f'layers_{lyr}', 'ResNeStSkipConnection_0', 'ConvBlock_0',
                    'Conv_0', 'kernel')
                add_bn(f'layer{b}.{i}.downsample.2',
                       (f'layers_{lyr}', 'ResNeStSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            lyr += 1

    lyr += 1
    pt2jax['fc.weight'] = ('params', f'layers_{lyr}', 'kernel')
    pt2jax['fc.bias'] = ('params', f'layers_{lyr}', 'bias')

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
