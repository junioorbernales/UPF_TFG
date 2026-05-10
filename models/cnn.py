import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

class CNNRegressor(nn.Module):
    def __init__(self, n_outputs=2):
        super().__init__()
        
        # Entrada inicial: (1, H, W)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bloques Residuales Profundos
        self.layer1 = ResidualBlock(32, 64)   # Captura detalles finos
        self.layer2 = ResidualBlock(64, 128, stride=2)  # Patrones temporales
        self.layer3 = ResidualBlock(128, 128) # Refinamiento
        self.layer4 = ResidualBlock(128, 256, stride=2) # Abstracción compleja
        self.layer5 = ResidualBlock(256, 256) # Bloque final de características
        
        # Global Pooling para ser invariante al tamaño de entrada
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4), # Aumentado ligeramente para regularizar la mayor complejidad
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_outputs),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.classifier(x)
    
# --- Nuevo Modelo Classifier ---
class CNNClassifier(nn.Module):
    def __init__(self, n_attack_classes=6, n_release_classes=4, dropout=0.4):
        super().__init__()
        
        # Extracción de características optimizada
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = ResidualBlock(32, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        self.layer5 = ResidualBlock(256, 256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Cabezal Attack (Ajustado a 6 clases por defecto)
        self.fc_attack = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_attack_classes)
        )
        
        # Cabezal Release
        self.fc_release = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_release_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        return self.fc_attack(x), self.fc_release(x)