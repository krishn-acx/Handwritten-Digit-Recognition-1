# MNIST Digit Classification

A deep learning implementation for handwritten digit recognition using Convolutional Neural Networks (LeNet-5 architecture).

## Project Overview

This project implements a **LeNet-5 inspired CNN** to classify handwritten digits (0-9) from the MNIST dataset with **98.05% test accuracy**.

## Dataset

**MNIST Database**
- **Total:** 70,000 images (60k training, 10k testing)
- **Dimensions:** 28 × 28 pixels (grayscale)
- **Classes:** 10 (digits 0-9)
- **Split:** 48k training, 12k validation, 10k test

## Model Architecture

| Layer | Type | Output Shape |
|-------|------|--------------|
| 1 | Conv2D (5×5, 6 filters) | (24, 24, 6) |
| 2 | AveragePooling2D (2×2) | (23, 23, 6) |
| 3 | Conv2D (5×5, 16 filters) | (19, 19, 16) |
| 4 | AveragePooling2D (2×2) | (9, 9, 16) |
| 5 | Conv2D (5×5, 120 filters) | (5, 5, 120) |
| 6 | Flatten | (3000,) |
| 7 | Dense (84 units) | (84,) |
| 8 | Dense (10 units) | (10,) |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.05%** |
| Training Accuracy | 97.90% |
| Validation Accuracy | 97.64% |

**Training Settings:**
- Epochs: 15
- Batch Size: 128
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

## Key Features

- **Data Preprocessing:** Z-score normalization, one-hot encoding
- **Model Training:** Early stopping, validation monitoring
- **Evaluation:** Confusion matrix, classification report, visual predictions
- **Performance:** Minimal overfitting, excellent generalization

## Requirements

- Python 3.8+
- TensorFlow 2.5+
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## Troubleshooting

**Memory Issues:** Reduce batch size to 64 or 32

**Slow Training:** Use GPU acceleration or reduce epochs

**Low Accuracy:** Verify normalization and encoding are applied correctly

## References

- LeCun et al., "Gradient-based learning applied to document recognition" (1998)
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- TensorFlow: https://www.tensorflow.org/
