# models/tcn_small.py
import torch
import torch.nn as nn

class TCNRegressorSmall(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=2, dropout=0.3):
        super().__init__()
        
        # Downsampling muy agresivo
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, 8, kernel_size=9, stride=2, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
            #nn.AdaptiveAvgPool1d(100),  # Forzar longitud pequeña
        )
        
        # Solo 1 bloque convolucional (no TCN complejo)
        self.features = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Clasificador tiny
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_outputs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.features(x)
        return self.classifier(x)