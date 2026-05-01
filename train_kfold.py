# train_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from data_utils.dataset import CompressorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
K_FOLDS = 5  # Validación cruzada con 5 folds
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'
MODEL_TYPE = "TCN"

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    # Cargar dataset completo (sin división train/val)
    full_dataset = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=32000)
    
    # K-Fold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Guardar resultados de cada fold
    fold_results = []
    all_folds_history = []  # ← Guardar historial de todos los folds
    
    print(f"\n🚀 Iniciando Validación Cruzada con {K_FOLDS} folds")
    print(f"Total muestras: {len(full_dataset)}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*50}")
        print(f"📂 Fold {fold+1}/{K_FOLDS}")
        print(f"   Train: {len(train_idx)} muestras")
        print(f"   Val:   {len(val_idx)} muestras")
        
        # Crear subsets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Modelo
        if MODEL_TYPE == "TCN":
            #from models.tcn import TCNRegressor
            from models.tcn_small import TCNRegressorSmall as TCNRegressor
            model = TCNRegressor(n_inputs=2, n_outputs=2, dropout=0.3).to(DEVICE)
        else:
            #from models.lstm import LSTMRegressor
            from models.lstm_small import LSTMRegressorSmall as LSTMRegressor
            model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        # ← Historial para este fold
        fold_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = validate(model, val_loader, criterion, DEVICE)
            
            # ← Guardar en historial
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Época {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo de este fold
                torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping en época {epoch+1}")
                    break
        
        # ← Guardar historial de este fold
        pd.DataFrame(fold_history).to_csv(f'history_fold_{fold+1}.csv', index=False)
        all_folds_history.append(fold_history)
        fold_results.append(best_val_loss)
        print(f"   ✅ Mejor Val Loss: {best_val_loss:.4f}")
    
    # Resultados finales
    print(f"\n{'='*50}")
    print("📊 RESULTADOS FINALES")
    print(f"   Loss por fold: {[round(r, 4) for r in fold_results]}")
    print(f"   Media: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    # Guardar resultados resumen
    pd.DataFrame({'fold': range(1, K_FOLDS+1), 'val_loss': fold_results}).to_csv('kfold_results.csv', index=False)

    # ← Graficar curvas de todos los folds juntos
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(all_folds_history):
        plt.plot(history['val_loss'], label=f'Fold {i+1} Val', linestyle='--', alpha=0.7)
        plt.plot(history['train_loss'], label=f'Fold {i+1} Train', linestyle=':', alpha=0.5)
    
    plt.xlabel('Época')
    plt.ylabel('Loss (MAE)')
    plt.title(f'Curvas de Entrenamiento - Validación Cruzada {K_FOLDS} folds ({MODEL_TYPE})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'kfold_curves_{MODEL_TYPE.lower()}.png', dpi=300)
    print(f"\n📈 Gráfico guardado: 'kfold_curves_{MODEL_TYPE.lower()}.png'")

if __name__ == "__main__":
    main()