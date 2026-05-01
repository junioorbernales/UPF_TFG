# evaluate.py (versión corregida)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
#from models.tcn import TCNRegressor  # ← Usa la versión con downsampling (la que entrenaste)
from models.tcn_small import TCNRegressorSmall as TCNRegressor
from models.lstm import LSTMRegressor 
from tqdm import tqdm

def evaluate_model():
    # 1. Configuración de modelo
    MODEL_TYPE = "TCN"  # o "LSTM"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NORM_ATTACK = 30.0
    NORM_RELEASE = 1.2
    
    if MODEL_TYPE == "TCN":
        MODEL_PATH = 'best_model_tcn.pth'
        # Usar la ARQUITECTURA CON DOWNSAMPLING (con la que entrenaste)
        model = TCNRegressor(
            n_inputs=2, 
            n_outputs=2
            # No especifiques num_channels ni kernel_size, usa los defaults de la clase
        ).to(DEVICE)
        
        # Verificar estructura del modelo guardado
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        print(f"🔍 Claves en el checkpoint: {list(checkpoint.keys())[:5]}...")
        print(f"📦 Total de claves: {len(checkpoint)}")
        
    else:
        MODEL_PATH = 'best_model_lstm.pth'
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    METADATA_CSV = 'data_ready/metadata.csv'
    AUDIO_ROOT = 'data_ready'
    
    # 2. Cargar datos de validación
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val')
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 3. Cargar pesos con manejo flexible
    print(f"Evaluando arquitectura: {MODEL_TYPE}...")
    
    try:
        # Intentar carga estricta
        model.load_state_dict(checkpoint)
        print("✅ Pesos cargados correctamente (carga estricta)")
    except RuntimeError as e:
        print(f"⚠️ Error en carga estricta: {e}")
        print("\nIntentando carga flexible...")
        
        # Carga flexible: solo pesos compatibles
        model_dict = model.state_dict()
        compatible_weights = {}
        
        for name, param in checkpoint.items():
            if name in model_dict and model_dict[name].shape == param.shape:
                compatible_weights[name] = param
                print(f"  ✓ Cargado: {name}")
            else:
                if name in model_dict:
                    print(f"  ✗ Saltado: {name} (shape mismatch: {param.shape} vs {model_dict[name].shape})")
                else:
                    print(f"  ✗ Saltado: {name} (no existe en el modelo)")
        
        model_dict.update(compatible_weights)
        model.load_state_dict(model_dict, strict=False)
        print(f"\n✅ Cargados {len(compatible_weights)}/{len(checkpoint)} pesos")
    
    model.eval()
    
    reales_norm = []
    predicciones_norm = []

    print("Procesando archivos de validación...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            preds = model(x)
            
            reales_norm.append(y.squeeze().cpu().numpy())
            predicciones_norm.append(preds.squeeze().cpu().numpy())

    # Convertir a arrays
    reales_norm = np.array(reales_norm)
    predicciones_norm = np.array(predicciones_norm)

    # Des-normalización
    reales = reales_norm.copy()
    predicciones = predicciones_norm.copy()
    
    reales[:, 0] *= NORM_ATTACK
    predicciones[:, 0] *= NORM_ATTACK
    
    reales[:, 1] *= NORM_RELEASE
    predicciones[:, 1] *= NORM_RELEASE

    # 4. Crear gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Attack
    ax1.scatter(reales[:, 0], predicciones[:, 0], alpha=0.4, color='royalblue', label='Predicciones')
    ax1.plot([reales[:, 0].min(), reales[:, 0].max()], 
             [reales[:, 0].min(), reales[:, 0].max()], 'r--', label='Ideal')
    ax1.set_title(f'Extracción de ATTACK ({MODEL_TYPE} con downsampling)')
    ax1.set_xlabel('Valor Real (ms)')
    ax1.set_ylabel('Predicción (ms)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Release
    ax2.scatter(reales[:, 1], predicciones[:, 1], alpha=0.4, color='forestgreen', label='Predicciones')
    ax2.plot([reales[:, 1].min(), reales[:, 1].max()], 
             [reales[:, 1].min(), reales[:, 1].max()], 'r--', label='Ideal')
    ax2.set_title(f'Extracción de RELEASE ({MODEL_TYPE} con downsampling)')
    ax2.set_xlabel('Valor Real (s)')
    ax2.set_ylabel('Predicción (s)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    nombre_archivo = f'resultados_extraccion_{MODEL_TYPE.lower()}_downsample.png'
    plt.savefig(nombre_archivo, dpi=300)
    print(f"✅ Gráfico guardado como '{nombre_archivo}'")
    
    # 5. Métricas
    mae = np.mean(np.abs(reales - predicciones), axis=0)
    print(f"\n--- Métricas Finales {MODEL_TYPE} (MAE en Unidades Reales) ---")
    print(f"Error medio en Attack: {mae[0]:.4f} ms")
    print(f"Error medio en Release: {mae[1]:.4f} s")
    
    # Calcular R² (coeficiente de determinación)
    ss_res_attack = np.sum((reales[:, 0] - predicciones[:, 0]) ** 2)
    ss_tot_attack = np.sum((reales[:, 0] - np.mean(reales[:, 0])) ** 2)
    r2_attack = 1 - (ss_res_attack / ss_tot_attack)
    
    ss_res_release = np.sum((reales[:, 1] - predicciones[:, 1]) ** 2)
    ss_tot_release = np.sum((reales[:, 1] - np.mean(reales[:, 1])) ** 2)
    r2_release = 1 - (ss_res_release / ss_tot_release)
    
    print(f"R² Attack: {r2_attack:.4f}")
    print(f"R² Release: {r2_release:.4f}")

if __name__ == "__main__":
    evaluate_model()