import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from data_utils.dataset import CompressorClassifierDataset 
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-3 # Subido para aprovechar el Warmup y la CNN Residual
EPOCHS = 150
K_FOLDS = 5
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'
MODEL_TYPE = "CNN"  
MODEL_NAME = MODEL_TYPE.lower()

def train_epoch(model, loader, optimizer, criterion_a, criterion_r, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        out_a, out_r = model(inputs)
        
        loss_a = criterion_a(out_a, targets[:, 0])
        loss_r = criterion_r(out_r, targets[:, 1])
        
        # Ponderación refinada para Attack
        loss = (0.55 * loss_a) + (0.45 * loss_r)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion_a, criterion_r, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out_a, out_r = model(inputs)
            
            loss_a = criterion_a(out_a, targets[:, 0])
            loss_r = criterion_r(out_r, targets[:, 1])
            loss = (0.55 * loss_a) + (0.45 * loss_r)
            
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    full_dataset = CompressorClassifierDataset(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=32000, augmentation_prob=0.3)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    all_folds_history = [] 
    
    # Definición de pesos para Attack
    attack_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.5]).to(DEVICE)
    criterion_a = nn.CrossEntropyLoss(weight=attack_weights, label_smoothing=0.1)
    criterion_r = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🚀 Iniciando Clasificación Cruzada ({K_FOLDS} folds) - Modelo: {MODEL_TYPE}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*50}\n📂 Fold {fold+1}/{K_FOLDS}")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        from models.cnn import CNNClassifier
        model = CNNClassifier(n_attack_classes=6, n_release_classes=4).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        fold_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion_a, criterion_r, DEVICE)
            val_loss = validate(model, val_loader, criterion_a, criterion_r, DEVICE)
            
            scheduler.step(val_loss)
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"    Época {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_classifier_{MODEL_NAME}_fold_{fold+1}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping en época {epoch+1}")
                    break
        
        all_folds_history.append(fold_history)
        fold_results.append(best_val_loss)

    # (Resto del código de resumen y plotting se mantiene igual)
    summary_csv = f'kfold_results_{MODEL_NAME}.csv'
    pd.DataFrame({'fold': range(1, K_FOLDS+1), 'best_val_loss': fold_results}).to_csv(summary_csv, index=False)
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(all_folds_history):
        plt.plot(history['val_loss'], label=f'Fold {i+1} Val', linestyle='--')
        plt.plot(history['train_loss'], label=f'Fold {i+1} Train', linestyle=':', alpha=0.5)
    plt.xlabel('Época')
    plt.ylabel('Loss (CrossEntropy)')
    plt.title(f'Curvas de Clasificación ({MODEL_TYPE})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'kfold_curves_classification_{MODEL_NAME}.png', dpi=300)
    print(f"\n✅ Entrenamiento completado. Archivos generados con sufijo '_{MODEL_NAME}'")

if __name__ == "__main__":
    main()