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
        
        # Apuntamos a la carpeta del stage (ej: data_ready/train)
        self.audio_root = Path(audio_root).resolve() / stage 
        self.target_sr = target_sr
        self.duration_samples = duration_samples
        
        self.audio_data = []
        self.params = []
        
        print(f"\n--- CARGANDO DATASET: {stage.upper()} ---")
        print(f"Ruta: {self.audio_root}")

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            filename = row['filename']
            
            # Buscamos el original (input) y el procesado (target)
            path_dry = self.audio_root / "input" / filename
            path_wet = self.audio_root / "target" / filename
            
            if path_dry.exists() and path_wet.exists():
                try:
                    # 1. Cargar pistas individuales
                    audio_dry, _ = librosa.load(str(path_dry), sr=self.target_sr, mono=True)
                    audio_wet, _ = librosa.load(str(path_wet), sr=self.target_sr, mono=True)
                    
                    # 2. Apilar en 2 canales: [0] = Dry, [1] = Wet
                    waveform = np.stack([audio_dry, audio_wet])
                    waveform_tensor = torch.from_numpy(waveform).float()
                    
                    # 3. ASEGURAR LONGITUD (Padding/Cropping)
                    # Si es más largo, recortamos
                    if waveform_tensor.shape[1] > self.duration_samples:
                        waveform_tensor = waveform_tensor[:, :self.duration_samples]
                    # Si es más corto, rellenamos con ceros
                    elif waveform_tensor.shape[1] < self.duration_samples:
                        padding = self.duration_samples - waveform_tensor.shape[1]
                        waveform_tensor = torch.nn.functional.pad(waveform_tensor, (0, padding))
                    
                    self.audio_data.append(waveform_tensor)
                    
                    # 4. Parámetros normalizados (Attack/30, Release/1.2)
                    self.params.append(torch.tensor([
                        row['attack'] / 30.0, 
                        row['release'] / 1.2
                    ], dtype=torch.float32))

                except Exception as e:
                    # print(f"Error en {filename}: {e}")
                    continue

        if len(self.audio_data) == 0:
            raise RuntimeError(f"No se pudo cargar ningún par de audios en {self.audio_root}")
        
        print(f"Cargadas {len(self.audio_data)} muestras con éxito.")

    def __getitem__(self, idx):
        return self.audio_data[idx], self.params[idx]

    def __len__(self):
        return len(self.audio_data)