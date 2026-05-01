import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
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


class TCNRegressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=2, dropout=0.5, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # Downsampling
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # TCN ligero
        layers = []
        in_ch = 64
        num_channels = [128, 128]
        
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size=self.kernel_size, dilation=dilation, dropout=dropout))
            in_ch = out_ch
        
        self.tcn = nn.Sequential(*layers)
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_outputs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.tcn(x)
        return self.classifier(x)