import torch
import pandas as pd
import librosa
import numpy as np  # Asegúrate de importar numpy
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

        audio_dry, _ = librosa.load(str(path_dry), sr=self.target_sr, mono=True)
        audio_wet, _ = librosa.load(str(path_wet), sr=self.target_sr, mono=True)

        # --- NORMALIZACIÓN DE PICO ---
        # Escalamos ambos canales para que su valor máximo absoluto sea 1.0
        # Esto elimina variaciones de volumen que no dependen de la compresión
        peak_dry = np.max(np.abs(audio_dry))
        if peak_dry > 1e-8:
            audio_dry = audio_dry / peak_dry
            
        peak_wet = np.max(np.abs(audio_wet))
        if peak_wet > 1e-8:
            audio_wet = audio_wet / peak_wet
        # -----------------------------

        if self.duration_samples is not None:
            if len(audio_dry) < self.duration_samples:
                audio_dry = librosa.util.pad_center(audio_dry, size=self.duration_samples)
                audio_wet = librosa.util.pad_center(audio_wet, size=self.duration_samples)
            elif len(audio_dry) > self.duration_samples:
                audio_dry = audio_dry[:self.duration_samples]
                audio_wet = audio_wet[:self.duration_samples]

        waveform = torch.stack([
            torch.from_numpy(audio_dry.copy()).float(),
            torch.from_numpy(audio_wet.copy()).float()
        ])

        params = torch.tensor([self.attack_list[idx], self.release_list[idx]], dtype=torch.float32)

        return waveform, params