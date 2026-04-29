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
        
        # Conexión residual para que el gradiente fluya
        self.res_connection = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res_connection(x) # Guardamos la entrada traducida
        out = self.conv(x)
        if self.conv.padding[0] > 0:
            out = out[:, :, :-self.conv.padding[0]]
        out = self.dropout(self.relu(self.bn(out)))
        return out + res  # <--- AQUÍ ESTÁ LA MAGIA: SUMAMOS LA ENTRADA

class TCNRegressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=2, num_channels=[64, 64, 128, 128, 256, 256, 512], kernel_size=5, dropout=0.2):
        super(TCNRegressor, self).__init__()
        
        layers = []
        in_ch = n_inputs
        # Incrementamos la dilatación exponencialmente: 1, 2, 4, 8...
        for i in range(len(num_channels)):
            dilation = 2 ** i 
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
        # Regresor final optimizado
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_outputs),
            nn.Sigmoid() # CRÍTICO: Mantiene la salida entre 0 y 1 para coincidir con tus targets normalizados
        )

    def forward(self, x):
        features = self.network(x)
        params = self.classifier(features)
        return params