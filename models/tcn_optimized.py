# models/tcn_optimized_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNRegressorOptimized(nn.Module):
    def __init__(self, n_inputs=3, n_outputs=2, dropout=0.2):
        super().__init__()
        
        # Filtro de suavizado para el canal GR (Canal 3)
        self.smooth_kernel_size = 65
        self.register_buffer('smooth_filter', torch.ones(1, 1, self.smooth_kernel_size) / self.smooth_kernel_size)
        
        # 1. Downsampling inicial
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, 32, kernel_size=11, stride=4, padding=5), # Stride 4 para mayor campo receptivo
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # 2. Bloques Dilatados: Dilatación agresiva para ver hasta 1.2s
        self.features = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, dilation=8, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, dilation=32, padding=96),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, dilation=64, padding=192),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # Reducimos a un vector pequeño pero informativo
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x: [Batch, 3, Samples] -> Dry, Wet, GR
        
        # Aplicamos logaritmo a la magnitud para comprimir el rango dinámico
        # Esto ayuda a que el Attack (cambios rápidos) destaque
        x = torch.log10(torch.abs(x) + 1e-4)
        
        # Suavizamos el canal de Gain Reduction (indice 2) para eliminar ruido
        gr_channel = x[:, 2:, :]
        b, c, s = gr_channel.shape
        gr_smooth = F.conv1d(gr_channel.view(b*c, 1, s), self.smooth_filter, padding=self.smooth_kernel_size//2)
        x[:, 2:, :] = gr_smooth.view(b, c, s)
        
        x = self.downsample(x) 
        x = self.features(x)
        return self.classifier(x)