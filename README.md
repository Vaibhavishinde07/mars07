# Speech Emotion Recognition using Pre-trained CNN Models

This repository implements a **Speech Emotion Recognition** system that classifies emotions from audio signals using three pre-trained CNN backbones: **ResNet101**, **VGG16**, and **VGG19**, on the RAVDESS dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)

   * Data Preprocessing
   * Feature Extraction
   * Model Architectures
4. [Installation & Usage](#installation--usage)

   * Dependencies
   * Running the Code
5. [Evaluation Results](#evaluation-results)

   * Overall Metrics
   * Detailed Per-Model Classification Reports
6. [Technical Implementation](#technical-implementation)
7. [Acknowledgements](#acknowledgements)

---

## Overview

The goal is to build an end-to-end pipeline that:

* Loads and preprocesses audio data.
* Extracts features (mel-spectrograms, MFCCs).
* Trains and evaluates pre-trained CNN models for emotion classification.

---

## Dataset

* **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

  * **7,356** recordings from **24** actors (12 female, 12 male)
  * Two neutral North American English statements
  * 8 emotion classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
  * 16-bit WAV, 48 kHz sampling rate

Data files are organized by modality (`Speech/`, `Song/`), with filenames following:

```
Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav
```

---

## Methodology

### 1. Data Preprocessing

* **Loading**: `librosa.load(..., sr=22050, duration=3.0)`
* **Normalization**: Scale amplitude to \[-1, 1]
* **Noise Reduction**: Optional spectral subtraction
* **Augmentation**:

  * Gaussian noise (0.3%)
  * Time stretching (0.85×–1.15×)
  * Pitch shifting (±3 semitones)
  * Applied to minority classes for balance

### 2. Feature Extraction

* **Mel-Spectrograms**: 64 mel bands → dB scale → normalization
* **MFCCs**: 13 coefficients + delta + delta-delta
* **RGB Conversion**: Stack spectrograms into 3 channels

### 3. Model Architectures

| Model     | Backbone    | Head Layers               | Optimizer (LR) |
| --------- | ----------- | ------------------------- | -------------- |
| ResNet101 | ResNet101V2 | GAP → 512 → 512 → 256 → 8 | Adam (1e-4)    |
| VGG16     | VGG16       | GAP → 512 → 512 → 256 → 8 | Adam (1e-4)    |
| VGG19     | VGG19       | GAP → 512 → 512 → 256 → 8 | Adam (1.55e-4) |

---

## Installation & Usage

### Dependencies

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib soundfile noisereduce scipy
```

### Code Snippet

Below is the core import and model-loading section for reference:

```python
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
# ... additional imports and model definitions ...
```

### Running the Pipeline

1. **Preprocess** the audio files to generate spectrogram images.
2. **Train** each model with:

   ```bash
   python train.py --model <ResNet101|VGG16|VGG19>
   ```
3. **Evaluate** using:

   ```bash
   python evaluate.py --model <ResNet101|VGG16|VGG19>
   ```

---

## Evaluation Results

### Overall Metrics

| Model     | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| --------- | -------- | -------- | ----------- | --------------- | ------------ |
| ResNet101 | 57.84%   | 57.78%   | 57.37%      | 57.21%          | 59.07%       |
| VGG16     | 59.06%   | 59.28%   | 57.56%      | 58.13%          | 61.93%       |
| VGG19     | 59.88%   | 60.26%   | 59.27%      | 59.70%          | 61.53%       |

### Detailed Classification Reports

**ResNet101**

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.6515    | 0.5733 | 0.6099   | 75      |
| Calm      | 0.6374    | 0.7733 | 0.6988   | 75      |
| Disgust   | 0.6047    | 0.6667 | 0.6341   | 39      |
| Fearful   | 0.6429    | 0.6000 | 0.6207   | 75      |
| Happy     | 0.5588    | 0.5067 | 0.5315   | 75      |
| Neutral   | 0.4600    | 0.6053 | 0.5227   | 38      |
| Sad       | 0.4167    | 0.3333 | 0.3704   | 75      |
| Surprised | 0.6047    | 0.6667 | 0.6341   | 39      |

**VGG16**

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.7385    | 0.6400 | 0.6857   | 75      |
| Calm      | 0.6500    | 0.6933 | 0.6710   | 75      |
| Disgust   | 0.6000    | 0.6923 | 0.6429   | 39      |
| Fearful   | 0.5684    | 0.7200 | 0.6353   | 75      |
| Happy     | 0.5231    | 0.4533 | 0.4857   | 75      |
| Neutral   | 0.5814    | 0.6579 | 0.6173   | 38      |
| Sad       | 0.3409    | 0.2000 | 0.2521   | 75      |
| Surprised | 0.6481    | 0.8974 | 0.7527   | 39      |

**VGG19**

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.7333    | 0.5867 | 0.6519   | 75      |
| Calm      | 0.6786    | 0.7600 | 0.7170   | 75      |
| Disgust   | 0.5455    | 0.6154 | 0.5783   | 39      |
| Fearful   | 0.6047    | 0.6933 | 0.6460   | 75      |
| Happy     | 0.5068    | 0.4933 | 0.5000   | 75      |
| Neutral   | 0.6190    | 0.6842 | 0.6500   | 38      |
| Sad       | 0.4211    | 0.3200 | 0.3636   | 75      |
| Surprised | 0.6667    | 0.7692 | 0.7143   | 39      |

**Key Findings**

* **Best Model**: VGG19 (59.88% accuracy, 60.26% macro F1)
* **Highest F1**: Calm (up to 0.717), Surprised (up to 0.753)
* **Most Challenging**: Sad (F1: 0.252–0.370)

---

## Technical Implementation

* **Languages**: Python 3.7+
* **Frameworks**: TensorFlow/Keras 2.x
* **Libraries**: librosa, scikit-learn, numpy, pandas, matplotlib
* **Hardware**: CUDA-compatible GPU recommended




