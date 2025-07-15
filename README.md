# 🐾 Animal Classification — Multi-Class Image Recognition with CNN

This repository presents a convolutional neural network (CNN) solution for multi-class image classification of animals using grayscale image data across 10 categories.

## 📂 Project Structure

- `train_dataset/datasets/Animals/{class}/` — training images organized by class  
- `guideline_code/guideline_code/datasets/Animals/{class}/` — validation images organized by class  
- `test_dataset/official_test_aug/` — test images for prediction  
- `model.ipynb` — Jupyter notebook for preprocessing, model training and prediction  
- `requirements.txt` — list of required packages  
- `README.md` — project overview and instructions  

## 🔧 Technologies Used

- Python 3  
- NumPy, OpenCV  
- TensorFlow / Keras  
- Jupyter Notebook  

## 📊 Dataset Description

The dataset contains grayscale animal images resized to 140x140 with pixel normalization to [0, 1]. There are 10 classes:
- `0` — butterflies  
- `1` — cats  
- `2` — chickens  
- `3` — cows  
- `4` — dogs  
- `5` — elephants  
- `6` — fishes  
- `7` — horses  
- `8` — lions  
- `9` — spiders  

Images are read using OpenCV, reshaped to `(140, 140, 1)`, and labels are one-hot encoded using `to_categorical`.

## 📈 Project Workflow

1. Load and preprocess training, validation, and test images  
2. Convert to grayscale, resize to 140x140, normalize, and reshape  
3. One-hot encode labels (for training and validation sets)  
4. Build a CNN model using Keras Sequential API  
5. Train the model on training data with validation monitoring  
6. Evaluate and use the model to predict classes for test images  

## 🚀 How to Run

Clone the repository:  
```
git clone https://github.com/SergKhachikyan/Animal_classification.git
```
```
cd Animal_classification
```
## Install dependencies:
```
pip install -r requirements.txt
```
