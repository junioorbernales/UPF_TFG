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
LR = 5e-4       # Subimos ligeramente el LR inicial para salir de la "zona plana"
EPOCHS = 100    # Aumentamos épocas para permitir mayor refinamiento
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets_params in tqdm(loader, desc="Training"):
        inputs, targets_params = inputs.to(device), targets_params.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs) 
        
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Cargando arquitectura: {MODEL_TYPE}...")
    
    if MODEL_TYPE == "TCN":
        from models.tcn import TCNRegressor 
        model = TCNRegressor(n_inputs=2, n_outputs=2).to(DEVICE)
    else:
        from models.lstm import LSTMRegressor 
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)

    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # NUEVO: Programador de Learning Rate
    # Si la pérdida no mejora en 5 épocas, reduce el LR a la mitad.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # Actualizamos el scheduler con la pérdida de validación
        scheduler.step(val_loss)
        
        # Obtener LR actual para el log
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Loss -> Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'best_model_regressor_{MODEL_TYPE.lower()}.pth'
            torch.save(model.state_dict(), save_path) 
            print(f"⭐ ¡Nuevo récord! Modelo guardado en {save_path}")

if __name__ == "__main__":
    main()