from functools import partial
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from flax.core import FrozenDict, freeze
from flax.traverse_util import unflatten_dict

from . import resnet
from .common import ModuleDef
from .splat import SplAtConv2d

try:
    import torch

    # https://github.com/pytorch/vision/issues/4156#issuecomment-894768539
    torch.hub._validate_not_a_forked_repo = lambda *_: True  # type: ignore
    torch_exists = True
except ImportError:
    torch_exists = False

PyTorchTensor = Any

# ResNet-D 50 from FastAI:
# github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py#L22
_RESNETD_URL = {50: 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'}

# ResNest from zhanghang1989/ResNeSt
# github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py#L16-L32
_RESNEST_URL_FMT = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest{}-{}.pth'  # noqa: E501
_RESNEST_SHA256 = {
    50: '528c19ca',
    101: '22405ba7',
    200: '75117900',
    269: '0cc87c48',
    '50_fast_2s1x64d': '44938639'
}
_RESNEST_URL = {
    size: _RESNEST_URL_FMT.format(size, _RESNEST_SHA256[size][:8])
    for size in _RESNEST_SHA256.keys()
}


def pretrained_resnet(
    size: int,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for ResNet ported from torch.hub.

    Args:
        size: 18, 34, 50, 101 or 152.
        state_dict: If provided, this state dict will be used over the
            pretrained torch.hub model. The keys must match the torch.hub resnet.

    Returns:
        Module Class and variables dictionary for Flax ResNet.
    """
    if size not in (18, 34, 50, 101, 152):
        raise ValueError('Ensure size is one of (18, 34, 50, 101, 152)')

    if state_dict is None:
        if not torch_exists:
            raise ImportError('Install `torch` to use this function.')

        state_dict = torch.hub.load('pytorch/vision:v0.10.0',
                                    f'resnet{size}',
                                    pretrained=True).state_dict()

    pt2jax: Dict[str, Sequence[str]] = {}
    add_bn = _get_add_bn(pt2jax)

    pt2jax['conv1.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))

    lyr = 2  # block_ind
    for b, n_blocks in enumerate(resnet.STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            for j in range(2 + (size >= 50)):
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


def pretrained_wide_resnet(
    size: int,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for Wide ResNet ported from torch.hub.

    Args:
        size: 50 or 101.
        state_dict: If provided, this state dict will be used over the
            pretrained torch.hub model. The keys must match the torch.hub wide
            resnet.

    Returns:
        Module Class and variables dictionary for Flax Wide ResNet.
    """
    if size not in (50, 101):
        raise ValueError('Ensure size is one of (50, 101)')

    if state_dict is None:
        if not torch_exists:
            raise ImportError('Install `torch` to use this function.')

        state_dict = torch.hub.load('pytorch/vision:v0.10.0',
                                    f'wide_resnet{size}_2',
                                    pretrained=True).state_dict()

    _, variables = pretrained_resnet(size, state_dict)
    model_cls = partial(getattr(resnet, f'WideResNet{size}'), n_classes=1000)
    return model_cls, variables


def pretrained_resnext(
    size: int,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for ResNeXt ported from torch.hub.

    Args:
        size: 50 or 101.
        state_dict: If provided, this state dict will be used over the
            pretrained torch.hub model. The keys must match the torch.hub resnext.

    Returns:
        Module Class and variables dictionary for Flax ResNeXt.
    """
    if size not in (50, 101):
        raise ValueError('Ensure size is one of (50, 101)')

    if state_dict is None:
        if not torch_exists:
            raise ImportError('Install `torch` to use this function.')

        state_dict = torch.hub.load(
            'pytorch/vision:v0.10.0',
            ('resnext50_32x4d' if size == 50 else 'resnext101_32x8d'),
            pretrained=True).state_dict()

    _, variables = pretrained_resnet(size, state_dict)
    model_cls = partial(getattr(resnet, f'ResNeXt{size}'), n_classes=1000)
    return model_cls, variables


def pretrained_resnetd(
    size: int,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for ResNet-D ported from Fast.AI.

    Fast.AI calls this model XResNet.

    Args:
        size: 50.
        state_dict: If provided, this state dict will be used over the
            pretrained fast.ai model. The keys must match the fastai xresnet.

    Returns:
        Module Class and variables dictionary for Flax ResNet-D.
    """
    if size not in _RESNETD_URL.keys():
        raise ValueError(f'Ensure `size` is one of {tuple(_RESNETD_URL.keys())}')

    if state_dict is None:
        if torch is None:
            raise ImportError('Install `torch` to use this function.')

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


def pretrained_resnest(
    size: int,
    fast: bool = False,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for ResNeSt ported from torch.hub.

    ResNeSt-Fast loads the `resnest50_fast_2s1x64d` variant.

    Args:
        size: 50, 101, 200, or 269.
        fast: whether to load ResNeSt-Fast or not.
        state_dict: If provided, this state dict will be used over the
            pretrained torch.hub model. The keys must match the torch.hub resnet.

    Returns:
        Module Class and variables dictionary for Flax ResNeSt.
    """
    if fast and size != 50:
        raise ValueError('Only ResNeSt 50 is supported with `fast=True`')
    elif size not in _RESNEST_URL.keys():
        raise ValueError(f'Ensure `size` is one of {tuple(_RESNEST_URL.keys())}')

    if state_dict is None:
        if not torch_exists:
            raise ImportError('Install `torch` to use this function.')

        url = _RESNEST_URL[size if not fast else '50_fast_2s1x64d']
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

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
                        splat_cls=partial(SplAtConv2d, match_reference=True),
                        avg_pool_first=fast)
    model_cls = getattr(resnet, f'ResNeSt{size}' if not fast else 'ResNeSt50Fast')
    model_cls = partial(model_cls, n_classes=1000, block_cls=block_cls)
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
