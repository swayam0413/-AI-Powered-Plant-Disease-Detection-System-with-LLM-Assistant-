# 🌱 AI-Powered Plant Disease Detection System (with LLM Assistant)

An intelligent web-based application that detects plant diseases from leaf images using Deep Learning and provides farmer-friendly disease explanations using a Large Language Model (LLM).

This project is developed as a Major Project for Advanced Software Development and focuses on Agriculture + AI + Computer Vision.

---

## 🚀 Project Overview

The Plant Disease Detection System allows users (farmers, students, researchers) to:

* Upload a leaf image
* Select crop type
* Detect plant disease using a trained AI model
* Get AI-generated explanation and treatment suggestions

The system combines:

* Image Processing (Computer Vision)
* Deep Learning (Transfer Learning)
* Large Language Model (LLM)
* Web Deployment (Flask)

---

## 🧠 Key Features

✅ Crop-wise disease prediction
✅ Image-based plant disease detection
✅ High accuracy using Transfer Learning (MobileNetV2)
✅ Farmer-friendly disease explanation using LLM
✅ Clean and interactive UI
✅ Real-time prediction via web interface
✅ Supports multiple crops (Apple, Tomato, Grape, etc.)

---

## 🏗️ System Architecture

```
User (Upload Image + Select Crop)
            ↓
      Flask Web App (Backend)
            ↓
  Preprocessing (160x160 + Normalization)
            ↓
 Deep Learning Model (MobileNetV2)
            ↓
 Predicted Disease + Confidence
            ↓
 LLM (Ollama/Gemini) generates:
   - Symptoms
   - Causes
   - Prevention
   - Treatment
            ↓
        Final Result on UI
```

---

## 🛠️ Tech Stack

### 💻 Frontend

* HTML5
* CSS3
* Bootstrap (UI Design)

### ⚙️ Backend

* Python
* Flask

### 🤖 AI & ML

* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Image Processing (PIL, NumPy)

### 🧾 LLM Integration

* Ollama (Local LLM)
* Gemini API (Optional)

### 📊 Dataset

* PlantVillage Dataset (39 Classes, 60K+ Images)

---

## 📁 Project Structure

```
Farmer_Inteligents/
│       
│
├── templates/
│   └── index.html             # Frontend UI
│
├── Train_model/               # Model Training Scripts
│     ├── app.py                       # Flask Backend 
|     ├── plant_disease_model_f.keras  # Trained AI Model 
|     ├── class_names_f.json 
| 
|--Model_Train
└── README.md                  # Project Documentation
```

---

## 📊 Model Details

* Model: MobileNetV2 (Transfer Learning)
* Input Size: 160 × 160 × 3
* Dataset: PlantVillage (Color Images)
* Classes: 39 Plant Disease Categories
* Training Platform: Kaggle GPU
* Accuracy: ~85% – 92%

---

## 🔬 Machine Learning Approach

1. Data Preprocessing (Resize + Normalization)
2. Data Augmentation (Flip, Rotation, Zoom)
3. Transfer Learning using MobileNetV2
4. Two-Phase Training:

   * Phase 1: Frozen Base Model
   * Phase 2: Fine-Tuning Top Layers
5. Model Optimization using EarlyStopping & ReduceLR

---

## 🌐 How to Run the Project (Local Setup)

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 🔹 Step 2: Create Virtual Environment

```bash
conda create -n mlenv python=3.10
conda activate mlenv
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

(Or manually)

```bash
pip install flask tensorflow pillow numpy ollama
```

### 🔹 Step 4: Add Model Files

Place these files in root folder:

* `plant_disease_model.keras`
* `class_names.json`

(Download from Kaggle output)

### 🔹 Step 5: Run Flask App

```bash
python app.py
```

### 🔹 Step 6: Open in Browser

```
http://127.0.0.1:5000
```

---

## 🤖 LLM Disease Explanation Feature

After prediction, the system sends the disease name to an LLM which generates:

* Disease summary
* Symptoms
* Causes
* Prevention methods
* Treatment suggestions (farmer-friendly)

Supported LLM Options:

* Ollama (Local AI)
* Gemini API (Cloud AI)

---

## 📸 User Workflow

1. Select Crop (e.g., Tomato, Apple, Grape)
2. Upload Leaf Image
3. Click “Predict Disease”
4. System shows:

   * Disease Name
   * Confidence Score
   * AI Explanation & Treatment

---

## ⚠️ Important Notes

* Image size during prediction must match training size (160×160)
* Use color dataset for better accuracy (not grayscale)
* Ensure model preprocessing uses MobileNetV2 preprocess function

## 🚀 Future Enhancements

* Mobile App Version (Android/iOS)
* Multi-language farmer support (Hindi/Regional)
* Real-time camera detection
* Disease severity estimation
* IoT integration for smart farming
* Cloud deployment (AWS/GCP)



## ⭐ If you like this project, give it a star on GitHub!
