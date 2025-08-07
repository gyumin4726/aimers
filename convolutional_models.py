import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    What is this for??
    This simply truncates chomp_size dots from the end of the sequence
    to make it the original length after Conv1d with
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, groups=1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):  # [2, 1, 500]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, groups=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, groups=groups)]
 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalConvNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dropout=0.2, stride=1, base_dilation=1):
        super(TemporalConvNet2D, self).__init__()

        assert out_channels % 4 == 0
        self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size,
                                           stride=stride, dilation=base_dilation*1))
        self.relu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(out_channels // 4, out_channels // 3, kernel_size,
                                           stride=stride, dilation=base_dilation*2))
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = weight_norm(nn.Conv2d(out_channels // 3, out_channels // 2, kernel_size,
                                           stride=stride*2, dilation=base_dilation*3))
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = weight_norm(nn.Conv2d(out_channels // 2, out_channels // 1, kernel_size,
                                           stride=stride*2, dilation=base_dilation*4))
        self.dropout4 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu, self.dropout1,
                                 self.conv2, self.relu, self.dropout2,
                                 self.conv3, self.relu, self.dropout3,
                                 self.conv4, self.relu, self.dropout4)

    def forward(self, x):
        return self.net(x)

class CNN1D2DDiscriminatorMultitask(nn.Module):
    def __init__(self, input_size, n_layers_1d, n_layers_2d, n_channel, n_channel_2d, class_count, kernel_size, dropout=0, groups=1):
        super(CNN1D2DDiscriminatorMultitask, self).__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers_1d
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, groups=groups)
        self.ccn = TemporalConvNet2D(1, n_channel_2d, kernel_size=3, dropout=dropout)
        self.n_channel_2d = n_channel_2d

        self.fault_type_head_fc1 = nn.Linear(self.n_channel_2d * 6 * 32 // 50, 128)
        self.fault_type_head_fc2 = nn.Linear(128, 32)
        self.fault_type_head_fc3 = nn.Linear(32, class_count)

        self.real_fake_head_fc1 = nn.Linear(self.n_channel_2d * 6 * 32 // 50, 64)
        self.real_fake_head_fc2 = nn.Linear(64, 1)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x, _, channel_last=True):
        common = self.tcn(x.transpose(1, 2) if channel_last else x).transpose(1, 2)
        common = common.unsqueeze(1)
        common = self.ccn(common)
        common = common.view(-1, 50, self.n_channel_2d * 6 * 32 // 50)
        
        type_logits = self.activation(self.fault_type_head_fc1(common))
        type_logits = self.activation(self.fault_type_head_fc2(type_logits))
        type_logits = self.fault_type_head_fc3(type_logits)

        real_fake_logits = self.activation(self.real_fake_head_fc1(common))
        real_fake_logits = torch.sigmoid(self.real_fake_head_fc2(real_fake_logits))

        return type_logits, real_fake_logits

    def zero_state(self, _):
        # just to make it compatible with LSTM architecture
        return torch.randn(1), torch.randn(1)


