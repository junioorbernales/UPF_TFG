import json
import os
from pedalboard import Pedalboard, Compressor
from pedalboard.io import AudioFile
import itertools

# 1. Setup paths and parameters
input_path = './data/raw_input/tono_10.wav'
output_dir = './data/processed/tono_10_compressor'
json_output_path = './data/processed/tono_10_processing_log.json'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

attack_values = [0.1, 0.3, 1, 3, 10, 30]  # ms
release_values = [0.1, 0.3, 0.6, 1.2]    # seconds (Note: Pedalboard uses ms)

# Initialize data list for JSON
processing_data = []

# 2. Process Audio
# We use itertools.product to get every combination of attack and release
for attack, release in itertools.product(attack_values, release_values):
    
    # Convert release to ms if your values are in seconds, as Pedalboard expects ms
    release_ms = release * 1000 
    
    with AudioFile(input_path) as f:
        # Define effect for this specific iteration
        board = Pedalboard([
            Compressor(
                threshold_db=-9, 
                ratio=2.0, 
                attack_ms=attack, 
                release_ms=release_ms
            )
        ])
        
        # Read the audio
        audio_data = f.read(f.frames)
        processed = board(audio_data, f.samplerate)
        
        # Define output filename
        filename = f'tono_10_attack_{attack}_release_{release}.wav'
        full_output_path = os.path.join(output_dir, filename)
        
        # Write the file
        with AudioFile(full_output_path, 'w', f.samplerate, f.num_channels) as o:
            o.write(processed)
            
        # Append metadata to our list
        processing_data.append({
            "filename": filename,
            "attack_ms": attack,
            "release_s": release,
            "threshold_db": -9,
            "ratio": 2.0
        })

# 3. Save to JSON
with open(json_output_path, 'w') as jf:
    json.dump(processing_data, jf, indent=4)

print(f"Processing complete. {len(processing_data)} files generated.")