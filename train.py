import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from tqdm import tqdm
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 3e-4       # Subimos ligeramente el LR inicial para salir de la "zona plana"
EPOCHS = 20    # Aumentamos épocas para permitir mayor refinamiento
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
    
    train_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='train', duration_samples=32000)
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val', duration_samples=32000)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 1. Crea listas para guardar la historia
    history = {
        'train_loss': [],
        'val_loss': []
    }

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # 2. Guarda los valores
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
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

    # 3. Al final de todas las épocas, guarda el historial a un CSV o JSON
    import pandas as pd
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    print("✅ Historial de entrenamiento guardado en training_history.csv")

    # 4. Opcional: Graficar la historia de pérdidas
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Curva de Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png') # Se guarda como imagen automáticamente
    print("📈 Gráfica de pérdida guardada como loss_curve.png")

if __name__ == "__main__":
    main()