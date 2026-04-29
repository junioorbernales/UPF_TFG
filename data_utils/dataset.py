import torch
import pandas as pd
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train', target_sr=16000, duration_samples=32000):
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['stage'] == stage]
        
        self.audio_root = Path(audio_root).resolve() / stage 
        self.target_sr = target_sr
        self.duration_samples = duration_samples
        
        # Guardamos solo las rutas y parámetros, no el audio (ahorro masivo de RAM)
        self.file_list = self.metadata['filename'].tolist()
        self.attack_list = (self.metadata['attack'] / 30.0).tolist()
        self.release_list = (self.metadata['release'] / 1.2).tolist()

        print(f"\n--- DATASET {stage.upper()} INICIALIZADO ---")
        print(f"Total archivos: {len(self.file_list)} | Ventana: {duration_samples} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        path_dry = self.audio_root / "input" / filename
        path_wet = self.audio_root / "target" / filename

        # Cargamos el audio justo en el momento de usarlo
        audio_dry, _ = librosa.load(str(path_dry), sr=self.target_sr, mono=True)
        audio_wet, _ = librosa.load(str(path_wet), sr=self.target_sr, mono=True)

        # Si el audio es más corto que lo que pide la red, hacemos padding (ceros)
        # Si es más largo, lo tomamos entero o lo ajustamos a la ventana
        if self.duration_samples is not None:
            # Padding si falta audio
            if len(audio_dry) < self.duration_samples:
                pad_len = self.duration_samples - len(audio_dry)
                audio_dry = librosa.util.pad_center(audio_dry, size=self.duration_samples)
                audio_wet = librosa.util.pad_center(audio_wet, size=self.duration_samples)
            # Recorte si sobra audio (puedes cambiar a recorte aleatorio si quieres)
            elif len(audio_dry) > self.duration_samples:
                audio_dry = audio_dry[:self.duration_samples]
                audio_wet = audio_wet[:self.duration_samples]

        waveform = torch.stack([
            torch.from_numpy(audio_dry).float(),
            torch.from_numpy(audio_wet).float()
        ])

        params = torch.tensor([self.attack_list[idx], self.release_list[idx]], dtype=torch.float32)

        return waveform, params