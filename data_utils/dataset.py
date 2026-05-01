import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train', target_sr=16000, duration_samples=32000):
        self.metadata = pd.read_csv(csv_path)
        
        # Si stage='all', no filtrar por stage
        if stage != 'all':
            self.metadata = self.metadata[self.metadata['stage'] == stage]
            self.audio_root = Path(audio_root).resolve() / stage
        else:
            # Para cross-validation, no filtramos y usamos carpeta base
            # Pero necesitamos ambas subcarpetas input/target
            self.audio_root = Path(audio_root).resolve()
        
        self.target_sr = target_sr
        self.duration_samples = duration_samples
        self.stage = stage  # Guardar stage para usarlo después
        
        self.file_list = self.metadata['filename'].tolist()
        self.attack_list = (self.metadata['attack'] / 30.0).tolist()
        self.release_list = (self.metadata['release'] / 1.2).tolist()

        print(f"\n--- DATASET {stage.upper()} INICIALIZADO ---")
        print(f"Total archivos: {len(self.file_list)} | Ventana: {duration_samples} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Para stage='all', las carpetas input/target están dentro de train/ y val/
        # Necesitamos obtener la etapa real del metadata
        if self.stage == 'all':
            # Obtener la etapa real de este archivo desde el metadata
            actual_stage = self.metadata.iloc[idx]['stage']
            path_dry = self.audio_root / actual_stage / "input" / filename
            path_wet = self.audio_root / actual_stage / "target" / filename
        else:
            path_dry = self.audio_root / "input" / filename
            path_wet = self.audio_root / "target" / filename

        audio_dry, _ = librosa.load(str(path_dry), sr=self.target_sr, mono=True)
        audio_wet, _ = librosa.load(str(path_wet), sr=self.target_sr, mono=True)

        # Normalización de pico
        peak_dry = np.max(np.abs(audio_dry))
        if peak_dry > 1e-8:
            audio_dry = audio_dry / peak_dry
            
        peak_wet = np.max(np.abs(audio_wet))
        if peak_wet > 1e-8:
            audio_wet = audio_wet / peak_wet

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


class CompressorDatasetWithAugmentation(CompressorDataset):
    def __init__(self, csv_path, audio_root, stage='train', 
                 duration_samples=32000, augmentation_prob=0.5):
        super().__init__(csv_path, audio_root, stage, duration_samples)
        self.augmentation_prob = augmentation_prob
    
    def __getitem__(self, idx):
        waveform, params = super().__getitem__(idx)
        
        if self.stage == 'train':
            if np.random.random() < self.augmentation_prob:
                # Ruido gaussiano
                noise_std = np.random.uniform(0.001, 0.01)
                noise = torch.randn_like(waveform) * noise_std
                waveform = waveform + noise
                
                # Shift temporal
                if np.random.random() > 0.7:
                    shift_pct = np.random.uniform(-0.03, 0.03)
                    shift = int(shift_pct * waveform.shape[-1])
                    waveform = torch.roll(waveform, shifts=shift, dims=-1)
                
                # Variación de ganancia
                if np.random.random() > 0.7:
                    gain = np.random.uniform(0.85, 1.15)
                    waveform = waveform * gain
        
        return waveform, params