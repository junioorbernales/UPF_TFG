import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from tqdm import tqdm
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets_params in tqdm(loader, desc="Training"):
        # inputs: [Batch, 2, samples] | targets_params: [Batch, 2]
        inputs, targets_params = inputs.to(device), targets_params.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs) 
        
        loss = criterion(predictions, targets_params) # Usamos el criterio definido en main
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets_params in tqdm(loader, desc="Validation"):
            inputs, targets_params = inputs.to(device), targets_params.to(device)
            
            predictions = model(inputs)
            loss = criterion(predictions, targets_params)
            running_loss += loss.item()
            
    return running_loss / len(loader)

def main():
    MODEL_TYPE = "TCN"  # "TCN" o "LSTM"
    
    # 1. Datasets y Loaders
    train_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='train')
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Inicializar Modelo
    print(f"Cargando arquitectura: {MODEL_TYPE}...")
    
    if MODEL_TYPE == "TCN":
        from models.tcn import TCNRegressor 
        model = TCNRegressor(n_inputs=2, n_outputs=2).to(DEVICE)
    else:
        from models.lstm import LSTMRegressor 
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)

    # 3. Loss y Optimizador
    # MSELoss es más estándar para regresión pura de parámetros
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Bucle de Entrenamiento
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Loss -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'best_model_regressor_{MODEL_TYPE.lower()}.pth'
            torch.save(model.state_dict(), save_path) 
            print(f"⭐ ¡Nuevo récord! Parámetros guardados en {save_path}")

if __name__ == "__main__":
    main()