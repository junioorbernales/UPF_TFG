import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from data_utils.dataset import CompressorDatasetWithAugmentation
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Reducido para mayor estabilidad con LSTM
LR = 1e-3 
EPOCHS = 60     # Aumentado para ver el efecto del Scheduler con tiempo
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'

# CAMBIA ESTO PARA ENTRENAR UNO U OTRO
MODEL_TYPE = "TCN" # Opciones: "TCN" o "LSTM"

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
    train_ds = CompressorDatasetWithAugmentation(METADATA_CSV, AUDIO_ROOT, stage='train', duration_samples=32000)
    val_ds = CompressorDatasetWithAugmentation(METADATA_CSV, AUDIO_ROOT, stage='val', duration_samples=32000)
    
    # num_workers=0 es más seguro en Windows para evitar errores de memoria compartida
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    history = {'train_loss': [], 'val_loss': []}

    print(f"\n🚀 Iniciando entrenamiento: {MODEL_TYPE}")
    
    if MODEL_TYPE == "TCN":
        #from models.tcn import TCNRegressor
        from models.tcn_small import TCNRegressorSmall as TCNRegressor
        model = TCNRegressor(n_inputs=2, n_outputs=2).to(DEVICE)
    else:
        from models.lstm import LSTMRegressor 
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=5e-3  # ← Aumentado de 1e-4 a 1e-3
    )
        
    # Programador: paciencia de 5 épocas para un entrenamiento más largo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 25
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Loss -> Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'best_model_{MODEL_TYPE.lower()}.pth'
            torch.save(model.state_dict(), save_path) 
            print(f"⭐ ¡Nuevo récord! Modelo guardado en {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 Early stopping en época {epoch+1}. No mejora desde hace {EARLY_STOPPING_PATIENCE} épocas.")
                break

    # Guardar historial específico
    pd.DataFrame(history).to_csv(f'history_{MODEL_TYPE.lower()}.csv', index=False)
    
    # Graficar y guardar con nombre del modelo
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Curva de Pérdida - {MODEL_TYPE}')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_{MODEL_TYPE.lower()}.png')
    print(f"✅ Proceso finalizado para {MODEL_TYPE}. Gráfica guardada.")

if __name__ == "__main__":
    main()