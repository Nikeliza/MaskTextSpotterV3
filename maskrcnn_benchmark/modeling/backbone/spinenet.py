import torch.nn as nn
import torch.nn.functional as F

from .resnet import Bottleneck, make_res_layer
from mmdet.ops import ConvModule
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
#from ..registry import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.
    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)
        
def make_res_layer(**kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (0, 1), False),
    (4, BasicBlock, (0, 1), False),
    (3, Bottleneck, (2, 3), False),
    (4, Bottleneck, (2, 4), False),
    (6, BasicBlock, (3, 5), False),
    (4, Bottleneck, (3, 5), False),
    (5, BasicBlock, (6, 7), False),
    (7, BasicBlock, (6, 8), False),
    (5, Bottleneck, (8, 9), False),
    (5, Bottleneck, (8, 10), False),
    (4, Bottleneck, (5, 10), True),
    (3, Bottleneck, (4, 10), True),
    (5, Bottleneck, (7, 12), True),
    (7, Bottleneck, (5, 14), True),
    (6, Bottleneck, (12, 14), True),
]

SCALING_MAP = {
    '49S': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level, block_fn, input_offsets, is_output):
    self.level = level
    self.block_fn = block_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  return [BlockSpec(*b) for b in block_specs]


class Resample(nn.Module):
    def __init__(self, in_channels, out_channels, scale, block_type, norm_cfg=dict(type="BN"), alpha=1.0):
        super(Resample, self).__init__()
        self.scale = scale
        new_in_channels = int(in_channels * alpha)
        if block_type == Bottleneck:
            in_channels *= 4
        self.squeeze_conv = ConvModule(in_channels, new_in_channels, 1, norm_cfg=norm_cfg)
        if scale < 1:
            self.downsample_conv = ConvModule(new_in_channels, new_in_channels, 3, padding=1, stride=2, norm_cfg=norm_cfg)
        self.expand_conv = ConvModule(new_in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def _resize(self, x):
        if self.scale == 1:
            return x
        elif self.scale > 1:
            return F.interpolate(x, scale_factor=self.scale, mode='nearest')
        else:
            x = self.downsample_conv(x)
            if self.scale < 0.5:
                new_kernel_size = 3 if self.scale >= 0.25 else 5
                x = F.max_pool2d(x, kernel_size=new_kernel_size, stride=int(0.5/self.scale), padding=new_kernel_size//2)
            return x

    def forward(self, inputs):
        feat = self.squeeze_conv(inputs)
        feat = self._resize(feat)
        feat = self.expand_conv(feat)
        return feat


class Merge(nn.Module):
    """Merge two input tensors"""
    def __init__(self, block_spec, norm_cfg, alpha, filter_size_scale):
        super(Merge, self).__init__()
        out_channels = int(FILTER_SIZE_MAP[block_spec.level] * filter_size_scale)
        if block_spec.block_fn == Bottleneck:
            out_channels *= 4
        self.block = block_spec.block_fn
        self.resample_ops = nn.ModuleList()
        for spec_idx in block_spec.input_offsets:
            spec = BlockSpec(*SPINENET_BLOCK_SPECS[spec_idx])
            in_channels = int(FILTER_SIZE_MAP[spec.level] * filter_size_scale)
            scale = 2**(spec.level - block_spec.level)
            self.resample_ops.append(
                Resample(in_channels, out_channels, scale, spec.block_fn, norm_cfg, alpha)
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.resample_ops)
        parent0_feat = self.resample_ops[0](inputs[0])
        parent1_feat = self.resample_ops[1](inputs[1])
        target_feat = parent0_feat + parent1_feat
        return target_feat


#@BACKBONES.register_module
class SpineNet(nn.Module):
    """Class to build SpineNet backbone"""
    def __init__(self,
                 arch,
                 in_channels=3,
                 output_level=[3, 4, 5, 6, 7], #[3, 4, 5, 6, 7]
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 zero_init_residual=True,
                 activation='relu'):
        super(SpineNet, self).__init__()
        self._block_specs = build_block_specs()[2:]
        self._endpoints_num_filters = SCALING_MAP[arch]['endpoints_num_filters']
        self._resample_alpha = SCALING_MAP[arch]['resample_alpha']
        self._block_repeats = SCALING_MAP[arch]['block_repeats']
        self._filter_size_scale = SCALING_MAP[arch]['filter_size_scale']
        self._init_block_fn = Bottleneck
        self._num_init_blocks = 2
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.zero_init_residual = zero_init_residual
        assert min(output_level) > 2 and max(output_level) < 8, "Output level out of range"
        self.output_level = output_level

        self._make_stem_layer(in_channels)
        self._make_scale_permuted_network()
        self._make_endpoints()

    def _make_stem_layer(self, in_channels):
        """Build the stem network."""
        # Build the first conv and maxpooling layers.
        self.conv1 = ConvModule(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the initial level 2 blocks.
        self.init_block1 = make_res_layer(
            self._init_block_fn,
            64,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
            self._block_repeats,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.init_block2 = make_res_layer(
            self._init_block_fn,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
            self._block_repeats,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def _make_endpoints(self):
        self.endpoint_convs = nn.ModuleDict()
        for block_spec in self._block_specs:
            if block_spec.is_output:
                in_channels = int(FILTER_SIZE_MAP[block_spec.level]*self._filter_size_scale) * 4
                self.endpoint_convs[str(block_spec.level)] = ConvModule(in_channels,
                                                                   self._endpoints_num_filters,
                                                                   kernel_size=1,
                                                                   norm_cfg=self.norm_cfg,
                                                                   act_cfg=None)

    def _make_scale_permuted_network(self):
        self.merge_ops = nn.ModuleList()
        self.scale_permuted_blocks = nn.ModuleList()
        for spec in self._block_specs:
            self.merge_ops.append(
                Merge(spec, self.norm_cfg, self._resample_alpha, self._filter_size_scale)
            )
            channels = int(FILTER_SIZE_MAP[spec.level] * self._filter_size_scale)
            in_channels = channels * 4 if spec.block_fn == Bottleneck else channels
            self.scale_permuted_blocks.append(
                make_res_layer(spec.block_fn,
                               in_channels,
                               channels,
                               self._block_repeats,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg)
            )

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, input):
        feat = self.maxpool(self.conv1(input))
        feat1 = self.init_block1(feat)
        feat2 = self.init_block2(feat1)
        block_feats = [feat1, feat2]
        output_feat = {}
        num_outgoing_connections = [0, 0]

        for i, spec in enumerate(self._block_specs):
            target_feat = self.merge_ops[i]([block_feats[feat_idx] for feat_idx in spec.input_offsets])
            # Connect intermediate blocks with outdegree 0 to the output block.
            if spec.is_output:
                for j, (j_feat, j_connections) in enumerate(
                        zip(block_feats, num_outgoing_connections)):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1
            target_feat = F.relu(target_feat, inplace=True)
            target_feat = self.scale_permuted_blocks[i](target_feat)
            block_feats.append(target_feat)
            num_outgoing_connections.append(0)
            for feat_idx in spec.input_offsets:
                num_outgoing_connections[feat_idx] += 1
            if spec.is_output:
                output_feat[spec.level] = target_feat

        return [self.endpoint_convs[str(level)](output_feat[level]) for level in self.output_level]
