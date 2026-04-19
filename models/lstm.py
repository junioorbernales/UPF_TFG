import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, n_outputs=2):
        super(LSTMRegressor, self).__init__()
        # input_size = 2 (canal 1: dry, canal 2: wet)
        # n_outputs = 2 (Attack, Release)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # La LSTM procesa la secuencia
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Capa de regresión final
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        # x: [Batch, Channels, Samples] -> Viene del DataLoader
        # Para la LSTM necesitamos: [Batch, Sequence_Length, Features]
        x = x.transpose(1, 2) 
        
        # h_n es el "último estado oculto", contiene la memoria de toda la secuencia
        _, (h_n, _) = self.lstm(x)
        
        # Tomamos el estado de la última capa de la LSTM
        last_hidden = h_n[-1] 
        
        # Predecimos los 2 parámetros
        params = self.regressor(last_hidden)
        return params # [Batch, 2]