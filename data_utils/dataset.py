import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train', target_sr=16000):
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['stage'] == stage]
        self.audio_root = audio_root
        self.target_sr = target_sr
        
        self.audio_data = []
        self.params = []
        
        print(f"Cargando y resampleando dataset de {stage} en RAM...")
        
        # Definimos el resampleador fuera del bucle para no recrearlo mil veces
        # Nota: Necesitamos saber la frecuencia original. Asumimos 44100 para el transform, 
        # pero lo manejamos dinámicamente abajo.
        
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            # 1. Cargar audio
            audio_path = os.path.join(self.audio_root, row['filename'])
            waveform, sr = torchaudio.load(audio_path)
            
            # 2. Resampling al vuelo (solo si es necesario)
            if sr != self.target_sr:
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # 3. Guardar en RAM
            # Usamos clone() para liberar memoria de objetos temporales
            self.audio_data.append(waveform.clone())
            
            # 4. Normalizar parámetros
            attack_norm = row['attack'] / 30.0
            release_norm = row['release'] / 1.2
            self.params.append(torch.tensor([attack_norm, release_norm], dtype=torch.float32))

    def __getitem__(self, idx):
        # Acceso instantáneo desde RAM, audio ya a 16kHz y parámetros normalizados
        return self.audio_data[idx], self.params[idx]

    def __len__(self):
        return len(self.audio_data)