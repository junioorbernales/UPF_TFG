import torch
import torch.nn as nn

class LSTMRegressorSmall(nn.Module):
    """
    LSTM con downsampling para procesamiento eficiente de audio.
    Reduce la longitud de la secuencia antes de pasar al LSTM.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, n_outputs=2, dropout=0.3):
        super(LSTMRegressorSmall, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 🔥 Downsampling - Reduce la secuencia drásticamente
        self.downsample = nn.Sequential(
            # 64,000 muestras → 16,000 (stride 4)
            nn.Conv1d(input_size, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 16,000 muestras → 4,000 (stride 4)
            nn.Conv1d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 4,000 muestras → 2,000 (stride 2)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Proyección a input_size para el LSTM
        self.feature_projection = nn.Conv1d(64, input_size, kernel_size=1)
        
        # LSTM ahora opera sobre una secuencia MUCHO más corta (~2000 pasos)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de regresión final
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Channels=2, Samples=64000]
        
        # 1. Downsampling: reduce la longitud temporal drásticamente
        x = self.downsample(x)  # [B, 64, 2000]
        
        # 2. Proyectar los features para que tengan input_size canales
        x = self.feature_projection(x)  # [B, input_size=2, 2000]
        
        # 3. Transponer para LSTM: [Batch, Sequence_Length, Features]
        x = x.transpose(1, 2)  # [B, 2000, 2]
        
        # 4. LSTM sobre secuencia corta
        _, (h_n, _) = self.lstm(x)
        
        # 5. Último estado oculto
        last_hidden = h_n[-1]  # [B, hidden_size]
        
        # 6. Regresión final
        params = self.regressor(last_hidden)
        
        return params