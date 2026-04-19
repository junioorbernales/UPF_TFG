import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import soundfile as sf

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train'):
        df = pd.read_csv(csv_path)
        self.metadata = df[df['stage'] == stage].reset_index(drop=True)
        self.audio_root = audio_root
        self.stage = stage

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        
        # Rutas a los archivos
        input_path = os.path.join(self.audio_root, self.stage, 'input', filename)
        target_path = os.path.join(self.audio_root, self.stage, 'target', filename)

        # 1. Leer audio con soundfile (devuelve numpy array)
        # sf.read devuelve (datos, samplerate)
        input_np, _ = sf.read(input_path)
        target_np, _ = sf.read(target_path)

        # 2. Convertir a tensores de PyTorch
        # Aseguramos que sean Float y tengan forma [1, Samples]
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_np).float().unsqueeze(0)

        # 3. Concatenar Dry y Wet para que el modelo vea ambos [2, Samples]
        x = torch.cat((input_tensor, target_tensor), dim=0)

        # 4. Los parámetros que queremos extraer (Targets)
        y = torch.tensor([row['attack'], row['release']], dtype=torch.float32)

        return x, y