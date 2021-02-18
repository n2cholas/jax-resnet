from functools import partial
from typing import Mapping, Tuple

from flax.core import freeze
from flax.traverse_util import unflatten_dict

from . import resnet
from .common import ModuleDef
from .splat import SplAtConv2d


def pretrained_resnet(size: int) -> Tuple[ModuleDef, Mapping]:
    '''Returns returns variables for ResNest from torch.hub.

    Args:
        size: 50, 101 or 152.

    Returns:
        Module Class and variables dictionary for Flax ResNet.
    '''
    try:
        import torch
    except ImportError:
        raise ImportError('Install `torch` to use this function.')

    if size not in (50, 101, 152):
        raise ValueError('Ensure size is one of (50, 101, 200, 269)')

    pt2jax = {}
    state_dict = torch.hub.load('pytorch/vision:v0.6.0',
                                f'resnet{size}',
                                pretrained=True).state_dict()

    def add_bn(pt_bn, jax_prefix):
        pt2jax[f'{pt_bn}.weight'] = ('params',) + jax_prefix + ('scale',)
        pt2jax[f'{pt_bn}.bias'] = ('params',) + jax_prefix + ('bias',)
        pt2jax[f'{pt_bn}.running_mean'] = ('batch_stats',) + jax_prefix + ('mean',)
        pt2jax[f'{pt_bn}.running_var'] = ('batch_stats',) + jax_prefix + ('var',)

    def bot(num):
        return f'ResNetBottleneckBlock_{num}'

    pt2jax['conv1.weight'] = ('params', 'ResNetStem_0', 'ConvBlock_0', 'Conv_0',
                              'kernel')
    add_bn('bn1', ('ResNetStem_0', 'ConvBlock_0', 'BatchNorm_0'))

    b_ind = 0  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            for j in range(3):
                pt2jax[f'layer{b}.{i}.conv{j+1}.weight'] = ('params', bot(b_ind),
                                                            f'ConvBlock_{j}', 'Conv_0',
                                                            'kernel')
                add_bn(f'layer{b}.{i}.bn{j+1}',
                       (bot(b_ind), f'ConvBlock_{j}', 'BatchNorm_0'))

            if f'layer{b}.{i}.downsample.0.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.0.weight'] = ('params', bot(b_ind),
                                                               'ResNetSkipConnection_0',
                                                               'ConvBlock_0', 'Conv_0',
                                                               'kernel')
                add_bn(f'layer{b}.{i}.downsample.1',
                       (bot(b_ind), 'ResNetSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            b_ind += 1

    pt2jax['fc.weight'] = ('params', 'Dense_0', 'kernel')
    pt2jax['fc.bias'] = ('params', 'Dense_0', 'bias')

    variables = {}
    for pt_name, jax_key in pt2jax.items():
        w = state_dict[pt_name].numpy()
        if w.ndim == 4:
            w = w.transpose((2, 3, 1, 0))
        elif pt_name == 'fc.weight':
            w = w.transpose()
        variables[jax_key] = w

    model_cls = partial(getattr(resnet, f'ResNet{size}'), n_classes=1000)
    return model_cls, freeze(unflatten_dict(variables))


def pretrained_resnest(size: int) -> Tuple[ModuleDef, Mapping]:
    '''Returns returns variables for ResNest from torch.hub.

    Args:
        size: 50, 101, 200, or 269.

    Returns:
        Module Class and variables dictionary for Flax ResNeSt.
    '''
    try:
        import torch
    except ImportError:
        raise ImportError('Install `torch` to use this function.')

    if size not in (50, 101, 200, 269):
        raise ValueError('Ensure size is one of (50, 101, 200, 269)')

    pt2jax_p, pt2jax_v = {}, {}
    BOT = 'ResNeStBottleneckBlock_{}'
    state_dict = torch.hub.load('zhanghang1989/ResNeSt',
                                f'resnest{size}',
                                pretrained=True).state_dict()

    def insert_bn(pt_bn, jax_prefix):
        pt2jax_p[f'{pt_bn}.weight'] = jax_prefix + ('scale',)
        pt2jax_p[f'{pt_bn}.bias'] = jax_prefix + ('bias',)
        pt2jax_v[f'{pt_bn}.running_mean'] = jax_prefix + ('mean',)
        pt2jax_v[f'{pt_bn}.running_var'] = jax_prefix + ('var',)

    # Stem
    pt2jax_p['conv1.0.weight'] = ('ResNeStStem_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    insert_bn('conv1.1', ('ResNeStStem_0', 'ConvBlock_0', 'BatchNorm_0'))
    pt2jax_p['conv1.3.weight'] = ('ResNeStStem_0', 'ConvBlock_1', 'Conv_0', 'kernel')
    insert_bn('conv1.4', ('ResNeStStem_0', 'ConvBlock_1', 'BatchNorm_0'))
    pt2jax_p['conv1.6.weight'] = ('ResNeStStem_0', 'ConvBlock_2', 'Conv_0', 'kernel')
    insert_bn('bn1', ('ResNeStStem_0', 'ConvBlock_2', 'BatchNorm_0'))

    b_ind = 0  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            pt2jax_p[f'layer{b}.{i}.conv1.weight'] = (BOT.format(b_ind), 'ConvBlock_0',
                                                      'Conv_0', 'kernel')
            insert_bn(f'layer{b}.{i}.bn1',
                      (BOT.format(b_ind), 'ConvBlock_0', 'BatchNorm_0'))

            # splat
            pt2jax_p[f'layer{b}.{i}.conv2.conv.weight'] = (BOT.format(b_ind),
                                                           'SplAtConv2d_0',
                                                           'ConvBlock_0', 'Conv_0',
                                                           'kernel')
            insert_bn(
                f'layer{b}.{i}.conv2.bn0',
                (BOT.format(b_ind), 'SplAtConv2d_0', 'ConvBlock_0', 'BatchNorm_0'))
            pt2jax_p[f'layer{b}.{i}.conv2.fc1.weight'] = (BOT.format(b_ind),
                                                          'SplAtConv2d_0',
                                                          'ConvBlock_1', 'Conv_0',
                                                          'kernel')
            pt2jax_p[f'layer{b}.{i}.conv2.fc1.bias'] = (BOT.format(b_ind),
                                                        'SplAtConv2d_0', 'ConvBlock_1',
                                                        'Conv_0', 'bias')
            insert_bn(
                f'layer{b}.{i}.conv2.bn1',
                (BOT.format(b_ind), 'SplAtConv2d_0', 'ConvBlock_1', 'BatchNorm_0'))
            pt2jax_p[f'layer{b}.{i}.conv2.fc2.weight'] = (BOT.format(b_ind),
                                                          'SplAtConv2d_0', 'Conv_0',
                                                          'kernel')
            pt2jax_p[f'layer{b}.{i}.conv2.fc2.bias'] = (BOT.format(b_ind),
                                                        'SplAtConv2d_0', 'Conv_0',
                                                        'bias')

            # rest
            pt2jax_p[f'layer{b}.{i}.conv3.weight'] = (BOT.format(b_ind), 'ConvBlock_1',
                                                      'Conv_0', 'kernel')
            insert_bn(f'layer{b}.{i}.bn3',
                      (BOT.format(b_ind), 'ConvBlock_1', 'BatchNorm_0'))

            # Downsample
            if f'layer{b}.{i}.downsample.1.weight' in state_dict:
                pt2jax_p[f'layer{b}.{i}.downsample.1.weight'] = (
                    BOT.format(b_ind), 'ResNeStSkipConnection_0', 'ConvBlock_0',
                    'Conv_0', 'kernel')
                insert_bn(f'layer{b}.{i}.downsample.2',
                          (BOT.format(b_ind), 'ResNeStSkipConnection_0', 'ConvBlock_0',
                           'BatchNorm_0'))

            b_ind += 1

    pt2jax_p['fc.weight'] = ('Dense_0', 'kernel')
    pt2jax_p['fc.bias'] = ('Dense_0', 'bias')

    variables: Mapping = {'params': {}, 'batch_stats': {}}
    params = variables['params']
    batch_stats = variables['batch_stats']
    for pt_name, jax_key in pt2jax_p.items():
        w = state_dict[pt_name].numpy()
        if w.ndim == 4:
            w = w.transpose((2, 3, 1, 0))
        elif pt_name == 'fc.weight':
            w = w.transpose()
        _nested_insert(params, jax_key, w)

    for pt_name, jax_key in pt2jax_v.items():
        w = state_dict[pt_name].numpy()
        _nested_insert(batch_stats, jax_key, w)

    block_cls = partial(resnet.ResNeStBottleneckBlock,
                        splat_cls=partial(SplAtConv2d, match_reference=True))
    model_cls = partial(getattr(resnet, f'ResNeSt{size}'),
                        n_classes=1000,
                        block_cls=block_cls)
    return model_cls, freeze(variables)


def _nested_lookup(d, keys):
    for k in keys:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def _nested_insert(d, keys, value):
    *prefix, key = keys
    d = _nested_lookup(d, prefix)
    d[key] = value
