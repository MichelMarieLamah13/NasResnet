import torch
import torch.nn as nn

OPS = {
    'none': lambda c, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda c, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda c, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda c, stride, affine: Identity() if stride == 1 else FactorizedReduce(c, c, affine=affine),
    'sep_conv_3x3': lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda c, stride, affine: SepConv(c, c, 7, stride, 3, affine=affine),
    'sep_conv_3x1': lambda c, stride, affine: SepConvTime(c, c, 3, stride, 1, affine=affine),
    'sep_conv_1x3': lambda c, stride, affine: SepConvFreq(c, c, 3, stride, 1, affine=affine),
    'dil_conv_3x3': lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda c, stride, affine: DilConv(c, c, 5, stride, 4, 2, affine=affine),
    'dil_conv_3x1': lambda c, stride, affine: DilConvTime(c, c, 3, stride, 2, 2, affine=affine),
    'dil_conv_1x3': lambda c, stride, affine: DilConvFreq(c, c, 3, stride, 2, 2, affine=affine),
    'basic_block': lambda c, stride, affine: BasicBlock(c_in=c, c_out=c, stride=stride, base_width=64),
    'conv_7x1_1x7': lambda c, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(c, c, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(c, affine=affine)
    ),
}


def conv3x3(c_in, c_out, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(c_in, c_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.bn1 = norm_layer(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(c_out, c_out)
        self.bn2 = norm_layer(c_out)
        self.stride = stride
        self.downsample = downsample
        if self.stride != 1 or c_in != c_out * BasicBlock.expansion:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, c_out * BasicBlock.expansion, stride),
                norm_layer(c_out * BasicBlock.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ReLUConvBN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConvTime(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConvTime, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0),
                      dilation=(dilation, 1), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConvFreq(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConvFreq, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding),
                      dilation=(1, dilation), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConvTime(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConvTime, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConvFreq(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConvFreq, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), groups=c_in,
                      bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, c_in, c_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.bn(out)
        return out
