import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train'):
        """
        csv_path: Ruta al metadata.csv generado
        audio_root: Ruta a la carpeta 'data_ready'
        stage: 'train' o 'val'
        """
        # Filtrar metadata por etapa (train/val)
        df = pd.read_csv(csv_path)
        self.metadata = df[df['stage'] == stage].reset_index(drop=True)
        self.audio_root = audio_root
        self.stage = stage

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        
        # Rutas a los archivos fragmentados
        input_path = os.path.join(self.audio_root, self.stage, 'input', filename)
        target_path = os.path.join(self.audio_root, self.stage, 'target', filename)

        # Cargar audio usando torchaudio (más rápido que librosa para entrenamiento)
        input_audio, _ = torchaudio.load(input_path)
        target_audio, _ = torchaudio.load(target_path)

        # Parámetros para FiLM (Attack y Release)
        # Los normalizamos o los pasamos como tensores
        params = torch.tensor([row['attack'], row['release']], dtype=torch.float32)

        return input_audio, target_audio, params