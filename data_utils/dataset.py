import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class CompressorDataset(Dataset):
    def __init__(self, csv_path, audio_root, stage='train', target_sr=16000, duration_samples=32000):
        self.metadata = pd.read_csv(csv_path)
        
        # Configuración de rutas según el stage
        if stage != 'all':
            self.metadata = self.metadata[self.metadata['stage'] == stage].reset_index(drop=True)
            self.audio_root = Path(audio_root).resolve() / stage
        else:
            self.audio_root = Path(audio_root).resolve()
        
        self.target_sr = target_sr
        self.duration_samples = duration_samples
        self.stage = stage
        
        self.file_list = self.metadata['filename'].tolist()
        # Normalización de targets (0.0 a 1.0)
        self.attack_list = (self.metadata['attack'] / 30.0).tolist()
        self.release_list = (self.metadata['release'] / 1.2).tolist()

        print(f"\n--- DATASET {stage.upper()} INICIALIZADO ---")
        print(f"Total archivos: {len(self.file_list)} | Ventana: {duration_samples} samples")

    def __len__(self):
        return len(self.file_list)

    def get_audio_pair(self, idx):
        """Carga y normaliza el par de audios Dry/Wet."""
        filename = self.file_list[idx]
        
        if self.stage == 'all':
            actual_stage = self.metadata.iloc[idx]['stage']
            path_dry = self.audio_root / actual_stage / "input" / filename
            path_wet = self.audio_root / actual_stage / "target" / filename
        else:
            path_dry = self.audio_root / "input" / filename
            path_wet = self.audio_root / "target" / filename

        audio_dry, _ = librosa.load(str(path_dry), sr=self.target_sr, mono=True)
        audio_wet, _ = librosa.load(str(path_wet), sr=self.target_sr, mono=True)

        # Normalización de pico unificada
        peak_dry = np.max(np.abs(audio_dry))
        if peak_dry > 1e-8:
            audio_dry = audio_dry / peak_dry
            audio_wet = audio_wet / peak_dry

        # Padding o Clipping para mantener tamaño constante
        if self.duration_samples is not None:
            if len(audio_dry) < self.duration_samples:
                audio_dry = librosa.util.pad_center(audio_dry, size=self.duration_samples)
                audio_wet = librosa.util.pad_center(audio_wet, size=self.duration_samples)
            else:
                audio_dry = audio_dry[:self.duration_samples]
                audio_wet = audio_wet[:self.duration_samples]
        
        return audio_dry, audio_wet

    def process_to_spectrogram(self, audio_dry, audio_wet, idx):
        """Convierte la diferencia de señales en un Espectrograma de Mel."""
        # Calculamos la reducción de ganancia (huella del compresor)
        gain_reduction = np.abs(audio_dry) - np.abs(audio_wet)

        # Generar el Mel Spectrogram
        # Salida aprox: [64 mels, 126 pasos de tiempo] para 32000 samples
        mel_spec = librosa.feature.melspectrogram(
            y=gain_reduction, 
            sr=self.target_sr, 
            n_mels=96,        # Subimos de 64 a 96 para más detalle
            hop_length=128,   # Bajamos de 256 a 128 (Doble resolución temporal)
            fmin=20,          # Rango audible
            fmax=8000
        )
        
        # Escala logarítmica (Decibelios)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalización min-max para la CNN (rango 0 a 1)
        mel_min = mel_spec_db.min()
        mel_max = mel_spec_db.max()
        mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min + 1e-6)

        # Convertir a tensor [Canal=1, H=64, W=126]
        spec_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)
        
        params = torch.tensor([self.attack_list[idx], self.release_list[idx]], dtype=torch.float32)

        return spec_tensor, params

    def __getitem__(self, idx):
        audio_dry, audio_wet = self.get_audio_pair(idx)
        return self.process_to_spectrogram(audio_dry, audio_wet, idx)


class CompressorDatasetWithAugmentation(CompressorDataset):
    def __init__(self, csv_path, audio_root, stage='train', 
                 target_sr=16000, duration_samples=32000, augmentation_prob=0.5):
        super().__init__(csv_path, audio_root, stage, target_sr, duration_samples)
        self.augmentation_prob = augmentation_prob
    
    def __getitem__(self, idx):
        # 1. Obtener audio original (ya viene normalizado de pico 1.0 por get_audio_pair)
        audio_dry, audio_wet = self.get_audio_pair(idx)
        
        # 2. Aplicar aumentación solo en entrenamiento y por probabilidad
        if (self.stage == 'train' or self.stage == 'all') and np.random.random() < self.augmentation_prob:
            
            # --- Aumentación A: Variación de Ganancia Global ---
            # Esto enseña al modelo que el Attack/Release es independiente del volumen de entrada
            gain = np.random.uniform(0.5, 1.0) # Bajamos volumen aleatoriamente
            audio_dry *= gain
            audio_wet *= gain

            # --- Aumentación B: Ruido Blanco (Dither) ---
            # Ayuda a que el modelo no se obsesione con silencios perfectos
            # El ruido debe ser muy tenue para no tapar la reducción de ganancia
            noise_std = np.random.uniform(0.0005, 0.002) 
            noise = np.random.normal(0, noise_std, audio_dry.shape).astype(np.float32)
            audio_dry += noise
            audio_wet += noise
            
            # --- Aumentación C: Inversión de Fase (Polaridad) ---
            # Un truco clásico en audio: al modelo no le debería importar si la onda va hacia arriba o abajo
            if np.random.random() < 0.5:
                audio_dry *= -1
                audio_wet *= -1
        
        # 3. Procesar a espectrograma
        return self.process_to_spectrogram(audio_dry, audio_wet, idx)