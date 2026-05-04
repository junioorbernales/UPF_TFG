# evaluate_kfold.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from data_utils.dataset import CompressorDataset
from tqdm import tqdm
import os

# --- Configuración (debe coincidir con train_kfold.py) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
K_FOLDS = 5
AUDIO_ROOT = 'data_ready'
METADATA_CSV = 'data_ready/metadata.csv'
MODEL_TYPE = "CNN"  # Cambiar a "LSTM" si entrenaste con ese modelo
DURATION_SAMPLES = 32000
MODEL_SUFFIX = MODEL_TYPE.lower()  # Sufijo dinámico para archivos

# NOTA: Ya no usamos NORM_ATTACK/RELEASE fijos porque la escala es Log10(x+1)
# Mantengo las variables por estructura, pero la lógica de des-normalización cambia abajo.

def instantiate_model(model_type, device):
    """Crea la arquitectura exacta usada durante el entrenamiento."""
    if model_type == "TCN":
        #from models.tcn import TCNRegressor
        #from models.tcn_small import TCNRegressorSmall as TCNRegressor
        from models.tcn_optimized import TCNRegressorOptimized as TCNRegressor
        model = TCNRegressor(n_inputs=3, n_outputs=2, dropout=0.3).to(device)
    # Dentro del bucle de folds, donde eliges el modelo:
    if MODEL_TYPE == "CNN":
        from models.cnn import CNNRegressor # Asegúrate de que el nombre coincide con tu archivo
        model = CNNRegressor(n_outputs=2).to(DEVICE)
    else:
        from models.lstm_small import LSTMRegressorSmall as LSTMRegressor
        model = LSTMRegressor(input_size=2, n_outputs=2).to(device)
    return model

def load_model_weights(model, checkpoint_path, device):
    """Carga pesos con manejo flexible por si hay ligeras diferencias de keys."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"   ⚠️ Carga estricta falló: {e}. Aplicando carga flexible...")
        model_dict = model.state_dict()
        compatible = {k: v for k, v in checkpoint.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        print(f"   ✅ Cargados {len(compatible)}/{len(checkpoint)} tensores.")
    return model

def evaluate_fold(model, loader, device):
    """Evalúa un modelo sobre un DataLoader y devuelve preds y targets."""
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Evaluando"):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y.cpu().numpy())
    return np.concatenate(preds_list, axis=0), np.concatenate(targets_list, axis=0)

def main():
    print(f"🚀 Iniciando evaluación K-Fold con {K_FOLDS} folds (Modelo: {MODEL_TYPE})")
    
    # 1. Recrear dataset y split EXACTO del entrenamiento
    full_dataset = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='all', duration_samples=DURATION_SAMPLES)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    all_preds_log = []
    all_targets_log = []
    fold_results = []
    
    print(f"📦 Total muestras en dataset: {len(full_dataset)}\n")
    
    # 2. Iterar por cada fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        model_path = f'best_model_fold_{fold+1}.pth'
        if not os.path.exists(model_path):
            print(f"⚠️  No se encontró {model_path}. Saltando fold {fold+1}.")
            continue
            
        print(f"{'='*40}")
        print(f"📂 Fold {fold+1}/{K_FOLDS} | Val: {len(val_idx)} muestras")
        
        # Subset y DataLoader solo para los datos de validación de este fold
        val_subset = Subset(full_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Instanciar y cargar modelo
        model = instantiate_model(MODEL_TYPE, DEVICE)
        model = load_model_weights(model, model_path, DEVICE)
        
        # Evaluar
        preds, targets = evaluate_fold(model, val_loader, DEVICE)
        all_preds_log.append(preds)
        all_targets_log.append(targets)
        
        # Métricas en escala logarítmica del fold
        mae_log = np.mean(np.abs(preds - targets), axis=0)
        fold_results.append({
            'fold': fold+1, 
            'n_samples': len(val_idx),
            'mae_attack_log': mae_log[0], 
            'mae_release_log': mae_log[1]
        })
        print(f"   ✅ MAE (log): Attack={mae_log[0]:.4f}, Release={mae_log[1]:.4f}")
        
    if not all_preds_log:
        print("❌ No se encontró ningún modelo para evaluar. Verifica los nombres de archivo.")
        return

# 3. Agregar resultados y des-normalizar linealmente [0, 1] -> Escala real
    all_preds_norm = np.concatenate(all_preds_log, axis=0)
    all_targets_norm = np.concatenate(all_targets_log, axis=0)
    
    all_preds = np.zeros_like(all_preds_norm)
    all_targets = np.zeros_like(all_targets_norm)

    # Des-normalización Attack: de [0, 1] a [0, 30] ms
    all_preds[:, 0] = all_preds_norm[:, 0] * 30.0
    all_targets[:, 0] = all_targets_norm[:, 0] * 30.0

    # Des-normalización Release: de [0, 1] a [0, 1.2] s
    all_preds[:, 1] = all_preds_norm[:, 1] * 1.2
    all_targets[:, 1] = all_targets_norm[:, 1] * 1.2

    # 4. Métricas globales en escala real
    mae = np.mean(np.abs(all_targets - all_preds), axis=0)
    
    ss_res_a = np.sum((all_targets[:, 0] - all_preds[:, 0]) ** 2)
    ss_tot_a = np.sum((all_targets[:, 0] - np.mean(all_targets[:, 0])) ** 2)
    r2_a = 1 - (ss_res_a / ss_tot_a) if ss_tot_a > 0 else 0.0
    
    ss_res_r = np.sum((all_targets[:, 1] - all_preds[:, 1]) ** 2)
    ss_tot_r = np.sum((all_targets[:, 1] - np.mean(all_targets[:, 1])) ** 2)
    r2_r = 1 - (ss_res_r / ss_tot_r) if ss_tot_r > 0 else 0.0

    print(f"\n{'='*40}")
    print(f"📊 RESULTADOS GLOBALES K-FOLD ({MODEL_TYPE.upper()})")
    print(f"   MAE Attack : {mae[0]:.4f} ms")
    print(f"   MAE Release: {mae[1]:.4f} s")
    print(f"   R² Attack  : {r2_a:.4f}")
    print(f"   R² Release : {r2_r:.4f}")

    # 5. Guardar métricas por fold y predicciones (nombres dinámicos)
    metrics_csv = f'kfold_metrics_{MODEL_SUFFIX}.csv'
    predictions_csv = f'kfold_predictions_{MODEL_SUFFIX}.csv'
    
    pd.DataFrame(fold_results).to_csv(metrics_csv, index=False)
    df_preds = pd.DataFrame({
        'real_attack_ms': all_targets[:, 0],
        'pred_attack_ms': all_preds[:, 0],
        'real_release_s': all_targets[:, 1],
        'pred_release_s': all_preds[:, 1]
    })
    df_preds.to_csv(predictions_csv, index=False)
    print(f"\n💾 Guardado: '{metrics_csv}' y '{predictions_csv}'")

    # 6. Generar gráficos consolidados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Attack
    ax1.scatter(all_targets[:, 0], all_preds[:, 0], alpha=0.5, color='royalblue', edgecolors='w', s=25)
    lim_a = [0, 30] # Rango fijo para Attack
    ax1.plot(lim_a, lim_a, 'r--', lw=2, label='Línea ideal')
    ax1.set_xlabel('Valor Real (ms)')
    ax1.set_ylabel('Predicción (ms)')
    ax1.set_title(f'Attack - Evaluación K-Fold ({MODEL_TYPE})')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Release
    ax2.scatter(all_targets[:, 1], all_preds[:, 1], alpha=0.5, color='forestgreen', edgecolors='w', s=25)
    lim_r = [0, 1.2] # Rango fijo para Release
    ax2.plot(lim_r, lim_r, 'r--', lw=2, label='Línea ideal')
    ax2.set_xlabel('Valor Real (s)')
    ax2.set_ylabel('Predicción (s)')
    ax2.set_title(f'Release - Evaluación K-Fold ({MODEL_TYPE})')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_name = f'kfold_evaluation_{MODEL_SUFFIX}.png'
    plt.savefig(plot_name, dpi=300)
    print(f"📈 Gráfico guardado como: '{plot_name}'")

if __name__ == "__main__":
    main()