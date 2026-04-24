import torch
import numpy as np
import pandas as pd
import os
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train', target_sr=16000, duration_samples=16000):
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['stage'] == stage]
        
        self.audio_root = Path(audio_root).resolve() / stage 
        self.target_sr = target_sr
        self.duration_samples = duration_samples # Por defecto 1 segundo a 16kHz
        
        self.audio_data = []
        self.params = []
        
        print(f"\n--- DEBUG DATASET ---")
        print(f"Buscando audios de {stage} en: {self.audio_root}")
        
        archivos_encontrados = list(self.audio_root.rglob("*.wav"))
        archivos_reales = {f.name.lower(): f for f in archivos_encontrados}
        
        print(f"Archivos .wav encontrados: {len(archivos_reales)}")
        
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            filename = str(row['filename']).lower()
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            if filename in archivos_reales:
                audio_path = archivos_reales[filename]
                try:
                    # 1. Carga con Librosa
                    waveform, _ = librosa.load(str(audio_path), sr=self.target_sr, mono=False)
                    
                    # 2. Asegurar 2 canales (si es mono, duplicamos)
                    if waveform.ndim == 1:
                        waveform = np.stack([waveform, waveform])
                    elif waveform.shape[0] > 2: # Si tiene más de 2, recortar
                        waveform = waveform[:2, :]
                    
                    waveform_tensor = torch.from_numpy(waveform).float()
                    
                    # 3. FORZAR LONGITUD (Crucial para el entrenamiento por batches)
                    # Si es más largo, recortamos
                    if waveform_tensor.shape[1] > self.duration_samples:
                        waveform_tensor = waveform_tensor[:, :self.duration_samples]
                    # Si es más corto, rellenamos con ceros (padding)
                    elif waveform_tensor.shape[1] < self.duration_samples:
                        padding = self.duration_samples - waveform_tensor.shape[1]
                        waveform_tensor = torch.nn.functional.pad(waveform_tensor, (0, padding))
                    
                    self.audio_data.append(waveform_tensor)
                    
                    # 4. Normalización de parámetros (Attack hasta 30ms, Release hasta 1.2s)
                    self.params.append(torch.tensor([
                        row['attack'] / 30.0, 
                        row['release'] / 1.2
                    ], dtype=torch.float32))

                except Exception:
                    continue

        if len(self.audio_data) == 0:
            raise RuntimeError(f"No se cargó nada en {stage}. Revisa los nombres en el CSV.")

    def __getitem__(self, idx):
        return self.audio_data[idx], self.params[idx]

    def __len__(self):
        return len(self.audio_data)