import torch
import torch.nn as nn

class TCNRegressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=2, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super(TCNRegressor, self).__init__()
        # n_inputs = 2 (canal 1: dry, canal 2: wet)
        # n_outputs = 2 (Attack, Release)
        
        layers = []
        in_ch = n_inputs
        for out_ch in num_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2) # Reducimos la resolución temporal a la mitad en cada capa
            ]
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
        # Capa final para estimar los 2 parámetros
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Colapsa la dimensión temporal a 1
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs) 
        )

    def forward(self, x):
        # x: [Batch, 2, Samples] -> dry y wet concatenados
        features = self.network(x)
        params = self.classifier(features)
        return params # [Batch, 2] -> Attack, Release predichos