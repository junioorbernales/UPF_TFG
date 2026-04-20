import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from tqdm import tqdm
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 3e-4 # Subimos un poco el LR, ya que ahora los targets son más pequeños (0 a 1)
EPOCHS = 50
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets_params in tqdm(loader, desc="Training"):
        inputs, targets_params = inputs.to(device), targets_params.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs) 
        
        # Si normalizaste en el dataset (/30 y /1.2), 
        # targets_params ya vienen en rango [0, 1]
        loss = criterion(predictions, targets_params) 
        
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
    MODEL_TYPE = "TCN" 
    
    train_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='train')
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val')
    
    # num_workers=0 suele evitar problemas de memoria/DLLs en Windows durante el entrenamiento
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Cargando arquitectura: {MODEL_TYPE}...")
    
    if MODEL_TYPE == "TCN":
        from models.tcn import TCNRegressor 
        model = TCNRegressor(n_inputs=2, n_outputs=2).to(DEVICE)
    else:
        from models.lstm import LSTMRegressor 
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)

    # Cambiamos a L1Loss (MAE). Al estar los datos entre 0 y 1, 
    # MSELoss hace los errores muy pequeños (ej: 0.1^2 = 0.01), lo que frena el aprendizaje.
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # Estas pérdidas ahora estarán en el rango de 0 a 1 (representando el % de error)
        print(f"Loss (Normalizada) -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'best_model_regressor_{MODEL_TYPE.lower()}.pth'
            torch.save(model.state_dict(), save_path) 
            print(f"⭐ ¡Nuevo récord! Modelo guardado en {save_path}")

if __name__ == "__main__":
    main()