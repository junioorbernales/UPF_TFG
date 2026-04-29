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
├── /data
│   ├── /raw_input     <-- Audios originales (Dry)
│   └── /processed     <-- Salidas del compresor con variaciones (Wet)
│
├── /data_ready        <-- Generada automáticamente (Ignorada por Git)
└── /data_utils
    └── prepare_dataset.py