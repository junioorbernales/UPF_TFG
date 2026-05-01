# models/tcn_original.py
import torch
import torch.nn as nn

class TCNBlockOriginal(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super(TCNBlockOriginal, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Conexión residual
        self.res_connection = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x):
        res = self.res_connection(x)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.dropout(self.relu(self.bn(out)))
        return out + res


class TCNRegressorOriginal(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=2, num_channels=[64, 64, 128, 128, 256, 256, 512], 
                 kernel_size=7, dropout=0.2):
        super(TCNRegressorOriginal, self).__init__()
        
        layers = []
        in_ch = n_inputs
        for i in range(len(num_channels)):
            dilation = 2 ** i 
            out_ch = num_channels[i]
            layers.append(TCNBlockOriginal(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.network(x)
        params = self.classifier(features)
        return params