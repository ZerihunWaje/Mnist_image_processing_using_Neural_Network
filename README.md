# MNIST Digit Classification with Keras
This project demonstrates how to build, train, evaluate, and visualize a simple neural network for handwritten digit classification using the MNIST dataset. The model is implemented with Keras and run on Google Colab.
## Project Overview

**Objective:** Classify grayscale images of handwritten digits (0–9) using a neural network.

**Dataset:** MNIST – 70,000 28x28 images (60,000 train, 10,000 test).

**Tools Used:** Python, Keras, TensorFlow backend, Matplotlib, Seaborn.

## Model Architecture

**Input layer:** 784 neurons (28x28 flattened image)

**Hidden layers:**

   Dense(128) + ReLU

   Dense(64) + ReLU

   Output layer: Dense(10) + Softmax

	model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')])

## ⚙️ How to Run
- Open the notebook in Google Colab.

- Run each cell sequentially to:

- Load and preprocess the dataset.

- Build and compile the model.

- Train the model.

- Evaluate its performance.

- Visualize predictions and misclassifications.
## 📊 Evaluation
- Loss Function: Categorical Crossentropy

- Optimizer: Adam

- Accuracy Achieved: ~97–98% on test set

## 📉 Sample Metrics
	
| Metric | Value |
| ----------- | ----------- |
|Training Accuracy | 	~98% |
| Test Accuracy | ~97% |
| Loss | Low (<0.1) |

## Key Learnings
- Effective preprocessing of image data

- Simple dense neural network design

- Model training and overfitting prevention

- Visualizing results to diagnose model behavior

## 💡 Future Improvements
- Use CNN instead of Dense layers

- Apply data augmentation

- Hyperparameter tuning (batch size, epochs, architecture)

- Deploy model as a web app

## 📚 References
- [Keras Documentation](https://keras.io/)

- [TensorFlow](https://www.tensorflow.org/)

- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
