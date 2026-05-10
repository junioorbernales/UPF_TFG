# train_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from data_utils.dataset import CompressorDataset
from data_utils.dataset import CompressorDatasetWithAugmentation
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 150
K_FOLDS = 5  # Validación cruzada con 5 folds
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'
MODEL_TYPE = "LSTM"

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))

def combined_loss(y_pred, y_true, weight_attack=0.35, weight_release=0.65):
    # Componente MSE: Castiga errores grandes (mejora R²)
    mse_attack = torch.nn.functional.mse_loss(y_pred[:, 0], y_true[:, 0])
    mse_release = torch.nn.functional.mse_loss(y_pred[:, 1], y_true[:, 1])
    
    # Componente L1: Ajuste fino y estabilidad
    l1_attack = torch.nn.functional.l1_loss(y_pred[:, 0], y_true[:, 0])
    l1_release = torch.nn.functional.l1_loss(y_pred[:, 1], y_true[:, 1])
    
    # Combinación 70% MSE / 30% L1
    loss_attack = 0.7 * mse_attack + 0.3 * l1_attack
    loss_release = 0.7 * mse_release + 0.3 * l1_release
    
    return (weight_attack * loss_attack) + (weight_release * loss_release)

def train_epoch(model, loader, optimizer, device, MODEL_TYPE):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        
        # --- CONDICIONAL DE LOSS ---
        if MODEL_TYPE == "LSTM":
            loss = combined_loss(predictions, targets)
        else:
            # --- SUSTITUCIÓN: Pérdida Ponderada ---
            # loss = criterion(predictions, targets) # ← Línea original sustituida
            
            # Calculamos pérdidas individuales para cada parámetro (normalizadas 0-1)
            loss_attack = log_cosh_loss(predictions[:, 0], targets[:, 0])
            loss_release = log_cosh_loss(predictions[:, 1], targets[:, 1])
            
            # Balanceamos: 35% peso al Attack, 65% peso al Release para forzar aprendizaje en la "cola"
            loss = (0.35 * loss_attack) + (0.65 * loss_release)
            # ----------------------------------------
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, device, MODEL_TYPE):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            
            # --- CONDICIONAL DE LOSS ---
            if MODEL_TYPE == "LSTM":
                loss = combined_loss(predictions, targets)
            else:
                # --- SUSTITUCIÓN: Pérdida Ponderada ---
                # loss = criterion(predictions, targets) # ← Línea original sustituida
                
                # Usamos la misma ponderación en validación para ser consistentes
                loss_attack = log_cosh_loss(predictions[:, 0], targets[:, 0])
                loss_release = log_cosh_loss(predictions[:, 1], targets[:, 1])
                loss = (0.35 * loss_attack) + (0.65 * loss_release)
                # ----------------------------------------
            
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    #full_dataset = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=32000)
    full_dataset = CompressorDatasetWithAugmentation(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=32000, augmentation_prob=0.5)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    all_folds_history = [] 
    
    print(f"\n🚀 Iniciando Validación Cruzada con {K_FOLDS} folds")
    print(f"Total muestras: {len(full_dataset)}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*50}")
        print(f"📂 Fold {fold+1}/{K_FOLDS}")
        print(f"   Train: {len(train_idx)} muestras")
        print(f"   Val:   {len(val_idx)} muestras")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        if MODEL_TYPE == "TCN":
            from models.tcn_optimized import TCNRegressorOptimized as TCNRegressor
            model = TCNRegressor(n_inputs=3, n_outputs=2, dropout=0.3).to(DEVICE)
        
        if MODEL_TYPE == "CNN":
            from models.cnn import CNNRegressor
            model = CNNRegressor(n_outputs=2).to(DEVICE)
        else:
            from models.lstm_small import LSTMRegressorSmall as LSTMRegressor
            model = LSTMRegressor(n_outputs=2).to(DEVICE)
        
        #criterion = log_cosh_loss
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        # --- Añadido: Scheduler ---
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        fold_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE, MODEL_TYPE)
            val_loss = validate(model, val_loader, DEVICE, MODEL_TYPE)
            
            # --- Añadido: Paso del scheduler ---
            scheduler.step(val_loss)
            
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Época {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping en época {epoch+1}")
                    break
        
        pd.DataFrame(fold_history).to_csv(f'history_fold_{fold+1}.csv', index=False)
        all_folds_history.append(fold_history)
        fold_results.append(best_val_loss)
        print(f"   ✅ Mejor Val Loss: {best_val_loss:.4f}")
    
    print(f"\n{'='*50}")
    print("📊 RESULTADOS FINALES")
    print(f"   Loss por fold: {[round(r, 4) for r in fold_results]}")
    print(f"   Media: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    pd.DataFrame({'fold': range(1, K_FOLDS+1), 'val_loss': fold_results}).to_csv('kfold_results.csv', index=False)

    plt.figure(figsize=(12, 6))
    for i, history in enumerate(all_folds_history):
        plt.plot(history['val_loss'], label=f'Fold {i+1} Val', linestyle='--', alpha=0.7)
        plt.plot(history['train_loss'], label=f'Fold {i+1} Train', linestyle=':', alpha=0.5)
    
    plt.xlabel('Época')
    plt.ylabel('Loss (Log-Cosh)')
    plt.title(f'Curvas de Entrenamiento - Validación Cruzada {K_FOLDS} folds ({MODEL_TYPE})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'kfold_curves_{MODEL_TYPE.lower()}.png', dpi=300)
    print(f"\n📈 Gráfico guardado: 'kfold_curves_{MODEL_TYPE.lower()}.png'")

if __name__ == "__main__":
    main()