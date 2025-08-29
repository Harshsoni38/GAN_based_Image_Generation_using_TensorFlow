# GAN-based Image Generation using TensorFlow

## 📌 Overview
This project implements a Generative Adversarial Network (GAN) using TensorFlow and Keras to generate synthetic images. The model is trained on the Fashion-MNIST dataset, which contains grayscale images of clothing items across 10 categories. The goal is to enable the generator network to learn the data distribution and produce new, realistic fashion images, while the discriminator learns to distinguish between real and generated samples.

---

## 📊 Dataset and Preprocessing
- Dataset: Fashion-MNIST (60,000 training images, 28x28 grayscale, 10 fashion categories)  
- Preprocessing steps:  
  - Normalization of pixel values to the range [0,1]  
  - Shuffling of the dataset to avoid order bias  
  - Mini-batch creation (batch size = 128) for stable training  
  - Prefetching to improve computational efficiency  

---

## 🏗️ Model Architectures

### 🔹 Generator
- Input: Random noise vector of dimension 128  
- Dense layer to expand noise and reshape into a 7x7x128 feature map  
- Series of upsampling and convolutional blocks with LeakyReLU activations  
- Final convolution layer with sigmoid activation producing a 28x28 grayscale image  
- Total parameters: ~2.15M  

### 🔹 Discriminator
- Input: 28x28 grayscale image (either real or generated)  
- Several convolutional layers with LeakyReLU activations and dropout regularization  
- Flattening and fully connected dense layer with sigmoid activation for binary classification (real vs fake)  
- Total parameters: ~1.11M  

---

## ⚙️ Training Setup
- Loss Function: Binary Cross-Entropy (BCE) for both generator and discriminator  
- Optimizer: Adam  
  - Generator learning rate: 0.0001  
  - Discriminator learning rate: 0.00001  
- Training Strategy:  
  - Generator attempts to fool the discriminator by producing realistic images  
  - Discriminator attempts to classify real vs generated images  
  - Both networks improve adversarially in a minimax game setup  
- Training implemented with a subclassed Keras model for custom training loops  
- Callback mechanism used to save generated images at the end of each epoch  

---

## 📈 Results
- Training logs show decreasing generator loss and oscillating discriminator loss, reflecting adversarial dynamics.  
- Generator progressively learns to create sharper and more realistic fashion images.  
- Example outcomes include generated synthetic T-shirts, trousers, and shoes that resemble real dataset samples.  

---

## 🚀 Key Highlights
- Built a complete GAN pipeline from scratch using TensorFlow and Keras  
- Covered data preprocessing, generator and discriminator architecture design, and custom training loop implementation  
- Achieved visually convincing synthetic fashion images after training  
- Demonstrates strong understanding of adversarial training, deep learning model design, and TensorFlow workflows  

---
