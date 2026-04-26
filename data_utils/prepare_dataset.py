import os
import json
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import re

def extract_params(filename):
    """Extrae attack y release del nombre del archivo."""
    
    match = re.search(
        r"attack_(\d*\.?\d+)_release_(\d*\.?\d+)",
        filename
    )

    if match:
        return float(match.group(1)), float(match.group(2))

    return None, None

def prepare_dataset(data_root, output_root, segment_sec=1.0, sr=44100):
    raw_dir = os.path.join(data_root, 'raw_input')
    processed_base = os.path.join(data_root, 'processed')
    
    # Estructura de salida
    for stage in ['train', 'val']:
        for folder in ['input', 'target']:
            os.makedirs(os.path.join(output_root, stage, folder), exist_ok=True)

    metadata_records = []
    
    # 1. Iterar por las carpetas de 'processed' (tono_01_compressor, etc.)
    proc_folders = [d for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))]
    
    print(f"Encontradas {len(proc_folders)} carpetas de procesamiento.")

    for folder in tqdm(proc_folders):
        # Extraer el ID del tono (ej: tono_01)
        tono_id = "_".join(folder.split("_")[:2]) 
        raw_file_path = os.path.join(raw_dir, f"{tono_id}.wav")
        
        if not os.path.exists(raw_file_path):
            continue

        # Cargar audio original (dry)
        x_full, _ = librosa.load(raw_file_path, sr=sr, mono=True)
        
        folder_path = os.path.join(processed_base, folder)
        # 2. Iterar por cada variación de attack/release en la carpeta
        for out_file in os.listdir(folder_path):
            if not out_file.endswith(".wav"): continue
            
            att, rel = extract_params(out_file)
            if att is None: continue

            # Cargar audio procesado (wet)
            y_full, _ = librosa.load(os.path.join(folder_path, out_file), sr=sr, mono=True)
            
            # Alinear longitudes
            min_len = min(len(x_full), len(y_full))
            hop = int(segment_sec * sr)
            
            # 3. Fragmentar
            for i in range(min_len // hop):
                start, end = i * hop, (i + 1) * hop
                chunk_x = x_full[start:end]
                chunk_y = y_full[start:end]
                
                # Definir si va a Train o Val (80/20 basado en el tono_id para evitar leak)
                stage = 'train' if (int(tono_id.split("_")[1]) % 5 != 0) else 'val'
                
                chunk_name = f"{tono_id}_att{att}_rel{rel}_s{i}.wav"
                
                # Guardar audios
                sf.write(os.path.join(output_root, stage, 'input', chunk_name), chunk_x, sr)
                sf.write(os.path.join(output_root, stage, 'target', chunk_name), chunk_y, sr)
                
                # Guardar metadatos para FiLM
                metadata_records.append({
                    'filename': chunk_name,
                    'stage': stage,
                    'attack': att,
                    'release': rel
                })

    # Guardar CSV de metadatos
    df = pd.DataFrame(metadata_records)
    df.to_csv(os.path.join(output_root, 'metadata.csv'), index=False)
    print(f"\nProceso finalizado. Metadatos guardados en {output_root}/metadata.csv")

if __name__ == "__main__":
    DATA_PATH = 'data' 
    OUTPUT_PATH = 'data_ready'

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No se encuentra la carpeta {DATA_PATH}.")
    else:
        prepare_dataset(
            data_root=DATA_PATH, 
            output_root=OUTPUT_PATH,
            segment_sec=2.0, # Correcto para ver el Release
            sr=16000,        # <--- ¡Añade esto! Coincide con tu entrenamiento
        )