import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRegressorSmall(nn.Module):
    def __init__(self, n_mels=96, hidden_size=128, num_layers=2, n_outputs=2, dropout=0.2):
        super(LSTMRegressorSmall, self).__init__()
        
        # 1. Extractor de características (CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 1)), 
        )

        self.lstm_input_size = (n_mels // 4) * 32
        
        # 2. Bloque Recurrente
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,      
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True 
        )

        # 3. Capa de Self-Attention (Attention Pooling)
        # Recibe la salida de la Bi-LSTM (hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 4. Regresor Final
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
            nn.Sigmoid() 
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param)
                    elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x):
        # x: [B, 1, n_mels, Time]
        
        # 1. CNN
        x = self.feature_extractor(x) 
        
        # 2. Reordenar para LSTM: [B, Time, Features]
        B, C, F_dim, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(B, T, -1)                    
        
        # 3. LSTM
        lstm_out, _ = self.lstm(x) # [B, T, hidden_size * 2]
        
        # 4. Cálculo de pesos de Atención
        # Generamos una puntuación para cada paso de tiempo
        attn_weights = self.attention(lstm_out) # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1) # Normalizamos (suman 1 en el eje T)
        
        # Suma ponderada: El modelo "decide" qué frames de tiempo mirar
        # [B, 1, T] @ [B, T, hidden_size * 2] -> [B, 1, hidden_size * 2]
        context_vector = torch.bmm(attn_weights.transpose(1, 2), lstm_out)
        context_vector = context_vector.squeeze(1) # [B, hidden_size * 2]
        
        # 5. Regresión
        return self.regressor(context_vector)

class LSTMClassifier(nn.Module):
    def __init__(self, n_mels=128, hidden_size=128, num_layers=2, 
                 n_attack_classes=6, n_release_classes=4, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        # 1. Extractor de características (CNN) 
        # Reduce la dimensión de frecuencia pero mantiene la temporal
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 1)), # Reduce Mels a la mitad
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 1)), # Reduce Mels a la cuarta parte
        )

        # Con n_mels=128 y dos MaxPool(2,1), la dimensión final de mels es 128/4 = 32
        self.lstm_input_size = (n_mels // 4) * 32
        
        # 2. Bloque Recurrente (Bi-LSTM)
        # Procesa la secuencia de frames del espectrograma
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,      
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True 
        )

        # 3. Capa de Self-Attention
        # Permite al modelo ignorar el silencio y enfocarse en el transitorio
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 4. Clasificadores Finales (Multi-head)
        self.fc_common = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        self.classifier_attack = nn.Linear(64, n_attack_classes)
        self.classifier_release = nn.Linear(64, n_release_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param)
                    elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x):
        # x: [B, 1, 128, T]
        
        # 1. CNN Feature Extraction
        x = self.feature_extractor(x) 
        
        # 2. Reshape para LSTM: [B, T, Features]
        B, C, F_dim, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(B, T, -1)                    
        
        # 3. Bi-LSTM
        lstm_out, _ = self.lstm(x) # [B, T, hidden_size * 2]
        
        # 4. Attention Pooling (Ponderación temporal)
        attn_weights = self.attention(lstm_out) # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1) 
        
        # Creamos el vector de contexto promediando los frames según su importancia
        context_vector = torch.bmm(attn_weights.transpose(1, 2), lstm_out)
        context_vector = context_vector.squeeze(1) 
        
        # 5. Clasificación
        common_features = self.fc_common(context_vector)
        out_attack = self.classifier_attack(common_features)
        out_release = self.classifier_release(common_features)
        
        return out_attack, out_release