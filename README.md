📂 Guía de Reproducción del Dataset
Este repositorio no incluye los archivos de audio debido a su gran tamaño. Para replicar los resultados del entrenamiento, sigue estos pasos:

1. Estructura de Carpetas
Asegúrate de que tu directorio local tenga la siguiente estructura:

Plaintext
/UPF_TFG
│
├── /data
│   ├── /raw_input    <-- Coloca aquí tus audios originales (Dry)
│   └── /processed    <-- Carpetas del compresor con variaciones (Wet)
│
├── /data_ready       <-- Se generará automáticamente (Ignorada por Git)
└── /data_utils
    └── prepare_dataset.py
2. Preparación de los Datos
Ejecuta el script de pre-procesamiento. Este script segmentará los audios en fragmentos de 2 segundos, realizará el resampleado a 16kHz y generará el archivo de metadatos necesario para el entrenamiento.

Bash
python data_utils/prepare_dataset.py
3. Entrenamiento
Una vez generada la carpeta data_ready, puedes lanzar el entrenamiento de la TCN:

Bash
python train.py
4. Notas Técnicas
Duración: Se han utilizado fragmentos de 2 segundos para asegurar que el modelo capture la envolvente completa de compresión (incluyendo el tiempo de Release máximo de 1.2s).

Entrada: El modelo recibe 2 canales (Canal 0: Dry, Canal 1: Wet).

Normalización: Los tiempos de Attack y Release se normalizan internamente entre 0 y 1 dividiendo por 30 y 1.2 respectivamente.