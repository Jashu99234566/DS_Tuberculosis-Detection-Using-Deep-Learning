# 🫱 Tuberculosis Detection from Chest X-rays

## 📚 Overview

This project builds a deep learning pipeline using **ResNet50** to classify chest X-ray images as either **Normal** or **Tuberculosis (TB)**. It includes model training, evaluation, and a real-time Streamlit web application for image upload and diagnosis.

---

## 🧩 Key Features

* 🧠 **Transfer Learning** with ResNet50 (ImageNet weights)
* 🔍 **Image Augmentation** for generalization
* ⚖️ **Class Weighting** to handle imbalance
* 📊 **Confusion Matrix & Classification Report**
* 🚮 **End-to-End Automation**: dataset split, training, evaluation
* 🌐 **Streamlit App** for X-ray classification
* 🔖 **Model Reuse** via `resnet_model.h5`

---

## ⚙️ Tech Stack

| Task               | Tools & Libraries                       |
| ------------------ | --------------------------------------- |
| **Modeling**       | TensorFlow / Keras                      |
| **Data Split**     | Scikit-learn, OS, shutil                |
| **Visualization**  | Matplotlib, Confusion Matrix, Streamlit |
| **Frontend**       | Streamlit Web UI                        |
| **Image Handling** | Pillow, ImageDataGenerator              |

---

## 📂 Project Structure

```
├── app.py                  # Streamlit app for live predictions
├── data_preparation.py     # Splits raw data into train/val/test
├── train_model.py          # ResNet50 fine-tuning and training
├── evaluate_model.py       # Model evaluation on test set
├── resnet_model.h5         # Saved trained model
└── TB_Chest_Xray_Data/     # Raw image dataset (with /TB and /Normal)
```

---

## 🚀 Getting Started

### ✅ Step 1: Install Requirements

```bash
pip install tensorflow streamlit pillow scikit-learn
```

### 📆 Step 2: Prepare Data

Ensure your data is in this format:

```
TB_Chest_Xray_Data/
├── TB/
└── Normal/
```

Then run:

```bash
python data_preparation.py
```

### 🔧 Step 3: Train Model

```bash
python train_model.py
```

### 📊 Step 4: Evaluate

```bash
python evaluate_model.py
```

### 🌐 Step 5: Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Prediction Output

* Upload an X-ray (PNG, JPG)
* Output:

  * ✅ Normal
  * ⚠️ Tuberculosis (TB)
* Displays confidence score from 0 to 1

---

## 🌟 Outcome

A complete image classification pipeline to assist in TB diagnosis from chest X-rays—ready for deployment or healthcare demo purposes.
