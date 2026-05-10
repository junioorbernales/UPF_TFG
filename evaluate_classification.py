# evaluate_classification.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from data_utils.dataset import CompressorClassifierDataset
from tqdm import tqdm
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
K_FOLDS = 5
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'
MODEL_TYPE = "LSTM"  # Cambiar a "LSTM" cuando sea necesario
MODEL_NAME = MODEL_TYPE.lower()
DURATION_SAMPLES = 32000

# Valores reales para las etiquetas de los ejes en los gráficos
ATTACK_LABELS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
RELEASE_LABELS = [0.1, 0.3, 0.6, 1.2]

def instantiate_model(model_type, device):
    """Instancia el modelo según el tipo definido."""
    if model_type == "CNN":
        from models.cnn import CNNClassifier
        model = CNNClassifier(n_attack_classes=len(ATTACK_LABELS), 
                              n_release_classes=len(RELEASE_LABELS)).to(device)
    elif model_type == "LSTM":
        from models.lstm_small import LSTMClassifier
        model = LSTMClassifier(n_attack_classes=len(ATTACK_LABELS), n_release_classes=len(RELEASE_LABELS)).to(device)
        #pass
    return model

def evaluate_fold(model, loader, device):
    model.eval()
    all_preds_a, all_preds_r = [], []
    all_targets_a, all_targets_r = [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Evaluando"):
            x, y = x.to(device), y.to(device)
            out_a, out_r = model(x)
            
            idx_a = torch.argmax(out_a, dim=1)
            idx_r = torch.argmax(out_r, dim=1)
            
            all_preds_a.append(idx_a.cpu().numpy())
            all_preds_r.append(idx_r.cpu().numpy())
            all_targets_a.append(y[:, 0].cpu().numpy())
            all_targets_r.append(y[:, 1].cpu().numpy())
            
    return (np.concatenate(all_preds_a), np.concatenate(all_preds_r), 
            np.concatenate(all_targets_a), np.concatenate(all_targets_r))

def plot_confusion_matrix(targets, preds, labels, title, filename):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción (Preset)')
    plt.ylabel('Real (Preset)')
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    print(f"🚀 Evaluación de Clasificación K-Fold (Modelo: {MODEL_TYPE})")
    
    full_dataset = CompressorClassifierDataset(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=DURATION_SAMPLES)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    final_targets_a, final_preds_a = [], []
    final_targets_r, final_preds_r = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        # Busca el archivo con el nombre dinámico del modelo
        model_path = f'best_classifier_{MODEL_NAME}_fold_{fold+1}.pth'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Saltando Fold {fold+1}: No se encontró {model_path}")
            continue
            
        print(f"📂 Procesando Fold {fold+1}...")
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
        
        model = instantiate_model(MODEL_TYPE, DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        pa, pr, ta, tr = evaluate_fold(model, val_loader, DEVICE)
        
        final_preds_a.extend(pa); final_targets_a.extend(ta)
        final_preds_r.extend(pr); final_targets_r.extend(tr)

    if len(final_targets_a) == 0:
        print("❌ No hay datos evaluados. Revisa que los archivos .pth existan.")
        return

    # Accuracy global
    acc_a = accuracy_score(final_targets_a, final_preds_a)
    acc_r = accuracy_score(final_targets_r, final_preds_r)

    print(f"\n📊 RESULTADOS GLOBALES ({MODEL_TYPE})")
    print(f"Accuracy Attack:  {acc_a:.4f} ({acc_a*100:.2f}%)")
    print(f"Accuracy Release: {acc_r:.4f} ({acc_r*100:.2f}%)")

    # Guardar Matrices de Confusión con nombre dinámico
    plot_confusion_matrix(final_targets_a, final_preds_a, ATTACK_LABELS, 
                          f'Matriz de Confusión: Attack ({MODEL_TYPE})', f'cm_attack_{MODEL_NAME}.png')
    plot_confusion_matrix(final_targets_r, final_preds_r, RELEASE_LABELS, 
                          f'Matriz de Confusión: Release ({MODEL_TYPE})', f'cm_release_{MODEL_NAME}.png')
    
    print(f"\n📈 Matrices guardadas como 'cm_attack_{MODEL_NAME}.png' y 'cm_release_{MODEL_NAME}.png'")

if __name__ == "__main__":
    main()