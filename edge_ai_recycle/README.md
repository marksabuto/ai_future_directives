# 🌐 Edge AI Recyclable Item Classifier

A lightweight image classification model trained to identify recyclable items using **TensorFlow Lite**, suitable for deployment on edge devices like Raspberry Pi.

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
- Colab or local machine with GPU support (optional)

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/marksabuto/AI-Future-Directives.git 
   cd AI-Future-Directives/edge_ai_recycle