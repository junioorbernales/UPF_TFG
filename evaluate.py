import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils.dataset import CompressorDataset
from models.tcn import TCNRegressor
from models.lstm import LSTMRegressor 
from tqdm import tqdm

def evaluate_model():
    # 1. Configuración de modelo
    MODEL_TYPE = "TCN"  # Cambia a "LSTM" manualmente aquí
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODEL_TYPE == "TCN":
        MODEL_PATH = 'best_model_regressor_tcn.pth'
        model = TCNRegressor(n_inputs=2, n_outputs=2).to(DEVICE)
    else:
        MODEL_PATH = 'best_model_regressor_lstm.pth'
        model = LSTMRegressor(input_size=2, n_outputs=2).to(DEVICE)

    METADATA_CSV = 'data_ready/metadata.csv'
    AUDIO_ROOT = 'data_ready'
    
    # 2. Cargar datos de validación
    val_ds = CompressorDataset(METADATA_CSV, AUDIO_ROOT, stage='val')
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 3. Cargar pesos
    print(f"Evaluando arquitectura: {MODEL_TYPE}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    reales = []
    predicciones = []

    print("Procesando archivos de validación...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            preds = model(x)
            
            reales.append(y.squeeze().cpu().numpy())
            predicciones.append(preds.squeeze().cpu().numpy())

    reales = np.array(reales)
    predicciones = np.array(predicciones)

    # 4. Crear Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico para ATTACK
    ax1.scatter(reales[:, 0], predicciones[:, 0], alpha=0.4, color='royalblue', label='Predicciones')
    ax1.plot([reales[:, 0].min(), reales[:, 0].max()], [reales[:, 0].min(), reales[:, 0].max()], 'r--', label='Ideal')
    ax1.set_title(f'Extracción de ATTACK ({MODEL_TYPE})')
    ax1.set_xlabel('Valor Real (ms)')
    ax1.set_ylabel('Predicción (ms)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Gráfico para RELEASE
    ax2.scatter(reales[:, 1], predicciones[:, 1], alpha=0.4, color='forestgreen', label='Predicciones')
    ax2.plot([reales[:, 1].min(), reales[:, 1].max()], [reales[:, 1].min(), reales[:, 1].max()], 'r--', label='Ideal')
    ax2.set_title(f'Extracción de RELEASE ({MODEL_TYPE})')
    ax2.set_xlabel('Valor Real (s)')
    ax2.set_ylabel('Predicción (s)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    nombre_archivo = f'resultados_extraccion_{MODEL_TYPE.lower()}.png'
    plt.savefig(nombre_archivo, dpi=300) # dpi=300 para calidad de impresión de TFG
    print(f"✅ Gráfico guardado como '{nombre_archivo}'")
    
    # Mostrar Error Medio
    mae = np.mean(np.abs(reales - predicciones), axis=0)
    print(f"\n--- Métricas Finales {MODEL_TYPE} (MAE) ---")
    print(f"Error medio en Attack: {mae[0]:.4f} ms")
    print(f"Error medio en Release: {mae[1]:.4f} s")

if __name__ == "__main__":
    evaluate_model()