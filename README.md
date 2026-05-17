# Handwritten Digit Recognition

Neural network trained on MNIST achieving **97% classification accuracy**, with a live inference module for custom handwritten input images.

---

## How it works

1. Loads and normalizes the MNIST dataset (70,000 samples, 28×28 grayscale images)
2. Trains a fully-connected neural network using TensorFlow/Keras
3. Evaluates against the 10,000-sample test set
4. Runs inference on custom digit images from the `digits/` folder

## Model architecture

```
Input (28×28) → Flatten → Dense(128, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)
```

Compiled with Adam optimizer and sparse categorical crossentropy loss. Trained for 4 epochs.

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **97%** |
| Training samples | 60,000 |
| Test samples | 10,000 |

## Setup

```bash
pip install tensorflow opencv-python numpy matplotlib
```

```bash
python main.py
```

Custom digit images go in the `digits/` folder, named `1.png` through `9.png` (white digit on black background).

## Stack

Python · TensorFlow · Keras · NumPy · OpenCV · Matplotlib · MNIST
