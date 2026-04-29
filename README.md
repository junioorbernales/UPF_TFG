# Modeling Audio Compression using Temporal Convolutional Networks (TCN) 🌊🎚️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![UPF](https://img.shields.io/badge/Institution-Universitat%20Pompeu%20Fabra-red.svg)](https://www.upf.edu/)

Este repositorio contiene el desarrollo mi **Trabajo de Final de Grado (TFG)**. El proyecto se centra en el modelado de procesadores de audio (compresión dinámica) mediante técnicas de **Deep Learning**, específicamente utilizando Redes Convolucionales Temporales (TCN).

---

## 📂 Guía de Reproducción del Dataset

Debido al gran tamaño de los archivos de audio, estos no se incluyen en el repositorio. Sigue estas instrucciones para preparar el entorno de entrenamiento:

### 1. Estructura de Directorios
Es fundamental mantener la siguiente jerarquía para que los scripts localicen los recursos correctamente:

```text
/UPF_TFG
│
├── /data                 <-- Audios base (No sincronizados en Git)
│   ├── /raw_input        <-- Señales originales sin procesar (Dry)
│   └── /processed        <-- Señales procesadas por hardware/software (Wet)
│
├── /data_ready           <-- Generado automáticamente (Ignorado por Git)
│   ├── /train            <-- Pares input/target para entrenamiento
│   └── /val              <-- Pares input/target para validación
│
├── /data_utils
│   ├── dataset.py        <-- Clase Loader (Carga dinámica y padding)
│   └── prepare_dataset.py <-- Script de segmentación y normalización
│
├── /models
│   ├── tcn.py            <-- Arquitectura Temporal Convolutional Network
│   └── lstm.py           <-- Arquitectura Long Short-Term Memory
│
├── train.py              <-- Script principal de entrenamiento y evaluación
└── README.md
```

### 2. Preparación de los Datos
Antes de iniciar el entrenamiento, los audios deben ser procesados para asegurar coherencia temporal y espectral. El script de preparación realiza el resampling a **16kHz** y segmenta los archivos en fragmentos de **2 segundos**.

```python
python data_utils/prepare_dataset.py
```
*Este comando generará la carpeta <u>/data_ready</u> y el archivo <u>metadata.csv</u> con las etiquetas de Attack y Release.*

### 3. Entrenamiento del Modelo
El entrenamiento está optimizado para comparar la eficacia de las redes **TCN** frente a las **LSTM**. Para lanzar el proceso con la configuración de 7 capas:

```python
python train.py
```
*Los resultados (Curvas de Loss y Scatter Plots) se guardarán automáticamente al finalizar las épocas definidas.*