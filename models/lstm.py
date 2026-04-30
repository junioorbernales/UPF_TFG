import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, n_outputs=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # He subido a 3 capas para darle más capacidad de "razonamiento"
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de regresión final con Sigmoid
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_outputs),
            nn.Sigmoid() # <--- CRÍTICO: Para coincidir con tus targets (0-1)
        )

    def forward(self, x):
        # x: [Batch, Channels, Samples] -> (8, 2, 32000)
        # Transponemos a: [Batch, Sequence_Length, Features] -> (8, 32000, 2)
        x = x.transpose(1, 2) 
        
        # h_n contiene el resumen de la secuencia después de ver los 2 segundos
        _, (h_n, _) = self.lstm(x)
        
        # h_n tiene forma [num_layers, batch, hidden_size]
        # Tomamos el estado oculto de la última capa
        last_hidden = h_n[-1] 
        
        # Predecimos Attack y Release
        params = self.regressor(last_hidden)
        return params