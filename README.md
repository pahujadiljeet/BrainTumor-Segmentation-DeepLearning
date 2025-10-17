# 🧠 Brain Tumor Segmentation using Deep Learning

[![Watch the video](https://img.youtube.com/vi/9wd21YIcHZA/0.jpg)](https://youtu.be/9wd21YIcHZA)

> 🎥 Click the image above to watch the full project demonstration on YouTube!

---

## 📘 Project Overview

This project focuses on **Brain Tumor Segmentation** using **Deep Learning (U-Net architecture)** applied to **MRI FLAIR images**.  
The goal is to accurately detect and highlight tumor regions in MRI scans for assisting radiologists and healthcare professionals in diagnosis.

The model achieves **pixel-level segmentation** and displays **side-by-side visual comparisons** of original and predicted tumor masks in an interactive **Streamlit web app**.

---

## 🚀 Key Features

✅ Preprocessing and normalization of MRI FLAIR images  
✅ Tumor segmentation using a **U-Net Deep Learning model**  
✅ Streamlit-based modern UI with:
   - 🎨 Thumbnail previews of all modalities  
   - 📊 Progress bar during prediction  
   - 🖼️ Side-by-side comparison of original vs predicted mask  
   - 💬 Footer with developer info and project credits  
✅ Supports multiple image uploads  
✅ Accurate and interactive segmentation results  

---

## 🧩 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Image Processing** | OpenCV, NumPy |
| **Visualization** | Matplotlib |
| **Web Interface** | Streamlit |
| **Dataset** | FLAIR MRI Scans (Brain Tumor Dataset) |

---

## 🧠 Project Architecture

1. **Data Preprocessing**  
   - Image resizing, normalization, and noise removal  
   - Preparing FLAIR MRI data for training/testing  

2. **Model Architecture (U-Net)**  
   - Encoder-Decoder CNN  
   - Skip connections for accurate boundary preservation  

3. **Training Phase**  
   - Loss: Binary Cross Entropy / Dice Coefficient  
   - Optimizer: Adam  
   - Evaluation metrics: Accuracy, IoU, Dice Score  

4. **Prediction & Visualization**  
   - Generates segmentation masks for uploaded images  
   - Streamlit displays results with comparison and progress bar  

---