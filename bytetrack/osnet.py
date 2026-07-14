from __future__ import division, absolute_import
import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

__all__ = [
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0',
    'build_osnet'
]

# Primary source: ImageNet-pretrained weights from KaiyangZhou/deep-person-reid
# (MIT license), re-hosted on this repository's GitHub releases for reliable
# direct downloads. Each entry is (url, sha256).
pretrained_urls = {
    'osnet_x1_0': (
        'https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_x1_0_imagenet.pth',
        'fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79'
    ),
    'osnet_x0_75': (
        'https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_x0_75_imagenet.pth',
        '37add23d2ebc7a22584203d6c22976fdda0f7f29c7b7d79257ac22ae20aec809'
    ),
    'osnet_x0_5': (
        'https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_x0_5_imagenet.pth',
        'f10ea9e26ee55b1a65cee3c6e88aa1ae105fe981af3e37692fa68a72fc64bf8e'
    ),
    'osnet_x0_25': (
        'https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_x0_25_imagenet.pth',
        'f54941a66bad4ddd07f2907f498c810ce639ce7a1abeaf2a151f8da118d84693'
    ),
    'osnet_ibn_x1_0': (
        'https://github.com/sweetlhare/BatchGpuByteTrack/releases/download/weights-v1.0/osnet_ibn_x1_0_imagenet.pth',
        '4e6fbaa3f5e31170aae76372a9ebbe36551e4011a4ed88b139d50bc98729e23a'
    ),
}

# Fallback source: original Google Drive links from KaiyangZhou/deep-person-reid
# (used only if the primary download fails; requires the optional `gdown` package)
gdrive_urls = {
    'osnet_x1_0':
    'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
    'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
    'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
    'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
    'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        IN=False,
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def _load_state_dict_partial(model, state_dict, source):
    """Load a state dict, keeping layers whose name or size doesn't match unchanged."""
    from collections import OrderedDict

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    if len(matched_layers) == 0:
        raise RuntimeError(
            f'No layers from "{source}" match the model — the checkpoint is '
            f'probably for a different architecture or corrupted.'
        )

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    logger.info('Loaded pretrained weights from "%s"', source)
    if discarded_layers:
        logger.debug(
            'Discarded layers due to unmatched keys or layer size: %s',
            discarded_layers
        )


def _download_pretrained_weights(key, cached_file):
    """Download pretrained weights, verifying sha256 for the primary source."""
    url, sha256 = pretrained_urls[key]
    try:
        torch.hub.download_url_to_file(url, cached_file, hash_prefix=sha256)
        return
    except Exception as e:
        logger.warning('Primary weight download failed (%s), trying Google Drive', e)

    try:
        import gdown
        gdown.download(gdrive_urls[key], cached_file, quiet=False)
    except ImportError:
        raise RuntimeError(
            f'Failed to download pretrained weights for "{key}" from {url} '
            f'and the fallback requires the optional "gdown" package '
            f'(pip install gdown). Alternatively download the file manually '
            f'and place it at "{cached_file}", or pass a local checkpoint '
            f'via build_osnet(..., checkpoint=...).'
        )

    import os
    if not os.path.exists(cached_file) or os.path.getsize(cached_file) == 0:
        raise RuntimeError(
            f'Failed to download pretrained weights for "{key}". Download '
            f'{url} manually and place it at "{cached_file}", or pass a local '
            f'checkpoint via build_osnet(..., checkpoint=...).'
        )


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Weights are cached under $TORCH_HOME/checkpoints (~/.cache/torch/checkpoints
    by default). Layers that don't match in name or size are kept unchanged.
    """
    import os

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    model_dir = os.path.join(_get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    cached_file = os.path.join(model_dir, key + '_imagenet.pth')

    if not os.path.exists(cached_file):
        _download_pretrained_weights(key, cached_file)

    state_dict = torch.load(cached_file, map_location='cpu', weights_only=True)
    _load_state_dict_partial(model, state_dict, cached_file)


##########
# Instantiation
##########
def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model


def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # medium size (width x0.75)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_75')
    return model


def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # tiny size (width x0.5)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_5')
    return model


def osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # very tiny size (width x0.25)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_25')
    return model


def osnet_ibn_x1_0(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    # standard size (width x1.0) + IBN layer
    # Ref: Pan et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV, 2018.
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ibn_x1_0')
    return model

def build_osnet(model_name='osnet_x0_5', pretrained=True, device='cuda', checkpoint=None, **kwargs):
    """
    Build OSNet re-identification model.

    Args:
        model_name (str): Model variant ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5',
            'osnet_x0_25', 'osnet_ibn_x1_0')
        pretrained (bool): Download and load ImageNet pretrained weights
            (ignored when `checkpoint` is given)
        device (str): Device to load model on ('cuda' or 'cpu'). Pass None to
            keep the model on CPU for manual placement.
        checkpoint (str): Optional path to a local .pth state-dict file; use this
            for fully offline setups
        **kwargs: Additional arguments for the model factory

    Returns:
        torch.nn.Module: OSNet model in eval mode
    """
    model_dict = {
        'osnet_x1_0': osnet_x1_0,
        'osnet_x0_75': osnet_x0_75,
        'osnet_x0_5': osnet_x0_5,
        'osnet_x0_25': osnet_x0_25,
        'osnet_ibn_x1_0': osnet_ibn_x1_0,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")

    # Build model
    model = model_dict[model_name](pretrained=pretrained and checkpoint is None, **kwargs)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        _load_state_dict_partial(model, state_dict, checkpoint)

    # Move to device (skip if device is None to allow manual device placement)
    if device is not None:
        model = model.to(device)
    model.eval()

    return model
