# Speech Emotion Recognition using Deep Learning with Pre-trained CNN Models

## Overview

This project implements a comprehensive speech emotion recognition system using state-of-the-art pre-trained convolutional neural networks (CNNs), including **ResNet101**, **VGG16**, and **VGG19**. The system processes audio signals from the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset to classify emotions with high accuracy, employing advanced feature extraction and data augmentation techniques.

## Dataset

The project utilizes the RAVDESS dataset, which contains:

* **Audio modalities**: Speech and Song
* **Emotion categories**: 8 classes (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
* **Performers**: 24 professional actors (12 female, 12 male)
* **Total samples**: Comprehensive audio recordings with varying intensities and statements

### Dataset Structure

Each filename in RAVDESS follows a systematic naming convention:

```
Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav
```

Fields encoded in the filename:

* **Modality**: Full-AV, Video-only, Audio-only
* **Vocal channel**: Speech, Song
* **Emotion**: One of the 8 emotional states
* **Intensity**: Normal, Strong
* **Statement**: Two distinct phrases
* **Repetition**: Multiple takes per condition
* **Actor**: Performer ID (gender-balanced)

## Methodology

### 1. Data Loading and Preprocessing

#### Advanced RAVDESS Data Loader

* **Filename Structure Decoding**: Parse RAVDESS naming convention
* **Metadata Extraction**: Extract gender, emotion, intensity, modality, and performer info
* **Dataset Compilation**: Organize audio samples with metadata labels

#### Audio Processing Pipeline

* **Sampling Rate**: Standardize to 22,050 Hz
* **Duration Normalization**: Trim or pad to 3-second segments
* **Signal Preprocessing**: Amplitude normalization and optional noise reduction
* **Padding/Truncation**: Ensure consistent length across samples

### 2. Feature Extraction System

#### Multi-Modal Feature Engineering

**Mel-Spectrogram Features (CNN Input)**

* 64 mel bands (0–8000 Hz)
* Min-max scaling to \[0, 1]
* 2D spectrograms compatible with image-based CNNs

**MFCC Temporal Features (Sequential Models)**

* 13 MFCC coefficients
* Statistical aggregations: mean, standard deviation, first/second derivatives
* 128-frame sequences → 52-dimensional feature vectors

**ResNet Input Preparation**

* Convert single-channel spectrograms to 3-channel RGB
* Resize to 224×224 pixels
* Normalize pixel values to \[0, 1]

### 3. Data Augmentation Strategy

#### Smart Audio Augmentation System

**Techniques:**

* Gaussian noise injection (0.3% intensity)
* Time stretching (0.85×–1.15×)
* Pitch shifting (±3 semitones)

**Intelligent Balancing:**

* Targeted augmentation for minority classes
* Quality control via temporary file management
* Balanced training set creation

### 4. Model Architectures

#### ResNet101 Implementation

```python
# Backbone: ResNet101V2 (ImageNet pretrained)
# Frozen convolutional layers
# Head: GAP → 512 → 512 → 256 → num_classes
# Dropout: 0.2 between dense layers
# Activation: ReLU (hidden), Softmax (output)
```

#### VGG16 Implementation

```python
# Backbone: VGG16 (ImageNet pretrained)
# Frozen convolutional layers
# Head: GAP → 512 → 512 → 256 → num_classes
# Dropout: 0.2
# Activation: ReLU (hidden), Softmax (output)
```

#### VGG19 Implementation

```python
# Backbone: VGG19 (ImageNet pretrained)
# Extended depth
# Head: GAP → 512 → 512 → 256 → num_classes
# Dropout: 0.2
# Activation: ReLU (hidden), Softmax (output)
```

### 5. Training Configuration

#### Optimization Strategy

* **Optimizer**: Adam
* **Learning Rate**: 0.0001 (0.0001155 for VGG19)
* **Batch Size**: 32
* **Epochs**: Up to 100
* **Class Weighting**: Balanced

#### Callbacks

* **EarlyStopping**: Patience=25 (monitor val\_accuracy)
* **ReduceLROnPlateau**: Factor=0.3, Patience=7, Min LR=1e-4
* **ModelCheckpoint**: Save best weights

### 6. Evaluation Methodology

* **Stratified Split**: 80% train (with augmentation), 20% test (no augmentation)
* **Metrics**:

  * Accuracy
  * Macro F1-score
  * Weighted F1-score
  * Precision, Recall
  * Per-class analysis

## Experimental Results

### Model Performance Comparison

| Model     | Accuracy   | Macro F1   | Weighted F1 | Macro Precision | Macro Recall |
| --------- | ---------- | ---------- | ----------- | --------------- | ------------ |
| **VGG19** | **0.5988** | **0.6026** | **0.5927**  | **0.5970**      | **0.6153**   |
| VGG16     | 0.5906     | 0.5928     | 0.5756      | 0.5813          | 0.6193       |
| ResNet101 | 0.5784     | 0.5778     | 0.5737      | 0.5721          | 0.5907       |

### Detailed Per-Class Performance

#### VGG19 (Best Model)

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.733     | 0.587  | 0.652    | 75      |
| Calm      | 0.679     | 0.760  | 0.717    | 75      |
| Disgust   | 0.545     | 0.615  | 0.578    | 39      |
| Fearful   | 0.605     | 0.693  | 0.646    | 75      |
| Happy     | 0.507     | 0.493  | 0.500    | 75      |
| Neutral   | 0.619     | 0.684  | 0.650    | 38      |
| Sad       | 0.421     | 0.320  | 0.364    | 75      |
| Surprised | 0.667     | 0.769  | 0.714    | 39      |

#### VGG16

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.738     | 0.640  | 0.686    | 75      |
| Calm      | 0.650     | 0.693  | 0.671    | 75      |
| Disgust   | 0.600     | 0.692  | 0.643    | 39      |
| Fearful   | 0.568     | 0.720  | 0.635    | 75      |
| Happy     | 0.523     | 0.453  | 0.486    | 75      |
| Neutral   | 0.581     | 0.658  | 0.617    | 38      |
| Sad       | 0.341     | 0.200  | 0.252    | 75      |
| Surprised | 0.648     | 0.897  | 0.753    | 39      |

#### ResNet101

| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.652     | 0.573  | 0.610    | 75      |
| Calm      | 0.637     | 0.773  | 0.699    | 75      |
| Disgust   | 0.605     | 0.667  | 0.634    | 39      |
| Fearful   | 0.643     | 0.600  | 0.621    | 75      |
| Happy     | 0.559     | 0.507  | 0.531    | 75      |
| Neutral   | 0.460     | 0.605  | 0.523    | 38      |
| Sad       | 0.417     | 0.333  | 0.370    | 75      |
| Surprised | 0.605     | 0.667  | 0.634    | 39      |

## Key Findings

* **Best Model**: VGG19 with 59.88% accuracy and 60.26% macro F1
* **Top Emotions**: Calm (F1 up to 0.717), Surprised (F1 up to 0.753)
* **Challenging**: Sad (F1 between 0.252–0.370)


## Dependencies

* TensorFlow/Keras 2.x
* Librosa, NumPy, Pandas, Scikit-learn
* Matplotlib, Seaborn, Plotly
* SoundFile, SciPy, NoiseReduce


