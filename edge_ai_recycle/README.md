# 🌐 Edge AI Recyclable Item Classifier

[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Edge%20AI-blue)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

A lightweight image classification model trained to identify recyclable items using **TensorFlow Lite**, suitable for deployment on edge devices like Raspberry Pi.

---

## 🚀 Project Overview

This project provides an efficient AI model for classifying recyclable materials (e.g., paper, plastic, metal, glass, trash, cardboard) in real time. The model is optimized for edge devices, enabling offline, low-latency operation for smart recycling bins, IoT sensors, and more.

## 🎯 Goal

To build an efficient AI model that can classify recyclable materials (e.g., paper, plastic, metal) with high accuracy while being optimized for real-time, offline use.

## 📁 Files Included

| File | Description |
|------|-------------|
| `train.ipynb` | Jupyter Notebook to train, evaluate, and convert the model |
| `model.h5` | Trained Keras model saved in HDF5 format |
| `recycle_classifier.tflite` | TensorFlow Lite version of the model for edge deployment |
| `README.md` | This file |

## 🔧 Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy, Matplotlib
- Jupyter Notebook
- (Optional) Colab or local machine with GPU support

## ⚡️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/marksabuto/AI-Future-Directives.git
   cd AI-Future-Directives/edge_ai_recycle
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. **Prepare your dataset:**
   - Organize images in subfolders by class under a root directory (e.g., `dataset/Paper`, `dataset/Plastic`, ...).
   - Update the `DATASET_PATH` in `train.ipynb` to your dataset location.

## 🏗️ Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet, feature extractor)
- **Input Size:** 224x224 RGB images
- **Output Classes:** Paper, Plastic, Metal, Glass, Trash, Cardboard
- **Layers:**
  - MobileNetV2 (frozen)
  - GlobalAveragePooling2D
  - Dense (softmax, 6 classes)

## ▶️ Usage

### 1. Training & Evaluation
Open `train.ipynb` in Jupyter or Colab and run all cells:
- Loads and augments data
- Trains MobileNetV2-based classifier
- Plots training/validation accuracy
- Saves model as `model.h5` and converts to `recycle_classifier.tflite`

### 2. Inference Example (TFLite)
```python
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="recycle_classifier.tflite")
interpreter.allocate_tensors()
# Prepare your input image as a (1, 224, 224, 3) numpy array, normalized to [0,1]
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_image)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print('Predicted class:', output.argmax())
```

## 📦 Model Conversion & Edge Deployment
- The notebook automatically converts the trained model to TensorFlow Lite format.
- Deploy `recycle_classifier.tflite` to edge devices (e.g., Raspberry Pi) using TFLite runtime or TensorFlow Lite Python API.
- For Raspberry Pi:
  - Install TFLite runtime: `pip install tflite-runtime`
  - Use the above inference code for predictions.

## 📊 Sample Results
- **Model:** MobileNetV2 + Dense (softmax)
- **Epochs:** 5 (default, can be increased)
- **Validation Accuracy:** ~85% (example, depends on dataset)
- ![Sample Accuracy Plot](docs/sample_accuracy.png) <!-- Add your plot or remove this line -->

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.

## 📄 License
This project is licensed under the MIT License.

---

**Contact:** [Your Name or Email]
