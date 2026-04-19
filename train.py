import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from tqdm import tqdm
import os

# --- Configuración (Hyperparameters) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets, params in tqdm(loader, desc="Training"):
        # Mover a GPU/CPU
        inputs = inputs.to(device)   # [Batch, 1, samples]
        targets = targets.to(device) # [Batch, 1, samples]
        params = params.to(device)   # [Batch, 2] (Attack, Release)

        optimizer.zero_grad()
        
        # PASO CLAVE: Pasamos audio Y parámetros
        outputs = model(inputs, params)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, params in tqdm(loader, desc="Validation"):
            inputs, targets, params = inputs.to(device), targets.to(device), params.to(device)
            outputs = model(inputs, params)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    # --- Parámetros de selección ---
    MODEL_TYPE = "TCN"  # Cambia a "LSTM" según necesites
    
    # 1. Datasets y Loaders
    # Asegúrate de que las rutas coincidan con tu salida de prepare_dataset.py
    train_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='train')
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Inicializar Modelo con lógica condicional
    print(f"Cargando arquitectura: {MODEL_TYPE}...")
    
    if MODEL_TYPE == "TCN":
        # Asegúrate de que el nombre del archivo sea tcn.py y la clase TCNBlock
        from models.tcn import TCNBlock 
        # Importante: n_params=2 para que el módulo FiLM sepa qué recibir
        model = TCNBlock(n_inputs=1, n_outputs=1, n_params=2).to(DEVICE)
    else:
        from models.lstm import LSTMModel # Cambia 'LSTM' por el nombre de tu clase
        # 3 canales: 1 audio + 1 attack + 1 release
        model = LSTMModel(input_size=3, hidden_size=64, num_layers=2).to(DEVICE)

    # 3. Loss y Optimizador
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Bucle de Entrenamiento
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        # Entrenar y validar
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Loss -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # GUARDADO CORREGIDO: state_dict() (estaba escrito state_state_dict)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Guardamos con el nombre del modelo para no sobrescribir si cambias de arquitectura
            save_path = f'best_model_{MODEL_TYPE.lower()}.pth'
            torch.save(model.state_dict(), save_path) 
            print(f"⭐ ¡Nuevo récord! Modelo guardado en {save_path}")

if __name__ == "__main__":
    main()