# 🌱 PlantGuard AI – Advanced Plant Disease Diagnostics

PlantGuard AI is an intelligent, full-stack web application designed to help farmers, agronomists, and home gardeners diagnose plant diseases instantly. The project combines **Deep Learning (PyTorch)** and **Large Language Models (Mistral AI)** to provide highly accurate disease classification alongside detailed, actionable, and beginner-friendly agricultural explanations.

---

## 🚀 Key Features

* **Advanced Neural Architecture:** Powered by **PlantDiseaseNet**, a custom PyTorch CNN model incorporating **Residual Blocks** and **CBAM (Convolutional Block Attention Module)** for spatial and channel-wise attention.
* **Exceptional Accuracy:** Achieves **97.09% accuracy** on the test set, with near-perfect healthy crop recognition.
* **Dual-Mode LLM Explanations:**
  - **Live Mistral API Integration:** Queries `open-mistral-7b` (with extended timeouts) to generate custom, detailed pathology reports.
  - **Fail-Safe Local Database:** If offline or the API key is not configured, the system automatically falls back to a comprehensive pre-written local database (`disease_database.json`) to guarantee detailed disease reports (symptoms, causes, prevention, treatment).
* **Premium Glassmorphic UI:**
  - Drag-and-drop file upload with instant leaf photo previews.
  - Animated visual scan laser indicating active diagnostics.
  - SVG circular confidence gauge (color-coded by confidence value).
  - Tabbed interactive section (Overview, Symptoms, Prevention, Treatment) that parses and renders markdown content beautifully.

---

## 📊 Model Evaluation Summary

The custom CNN was trained on **20,638 images** from the PlantVillage dataset across 15 distinct classes (covering Tomato, Potato, and Pepper Bell crops):

* **Total Parameters:** 11,394,519
* **Best Validation Accuracy:** 96.80%
* **Test Accuracy:** 97.09%
* **Test F1-Score:** 0.9710
* **Training Time:** 55.6 min (optimized for Kaggle GPU)

For a visual breakdown of convergence and per-class reports, refer to `training_history.png` and `confusion_matrix.png` in the notebook source folder.

---

## 📂 Project Structure

```
plant_disease_detection/
├── templates/
│   └── index.html                 # Premium Glassmorphic UI
├── app.py                         # Flask Backend & PyTorch inference logic
├── disease_database.json          # Fail-safe local disease database
├── plant_disease_model_final.pth  # Trained PyTorch Model weights (45MB)
├── Model_Train.ipynb              # Notebook source detailing training steps
├── .env.example                   # Template for environment configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🛠️ Local Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/swayam0413/-AI-Powered-Plant-Disease-Detection-System-with-LLM-Assistant-.git
cd -AI-Powered-Plant-Disease-Detection-System-with-LLM-Assistant-
```

### Step 2: Set Up Conda Environment
Use your conda `ai` environment or create a new one:
```bash
conda create -n ai python=3.11
conda activate ai
```

### Step 3: Install Dependencies
Run the following command to install the required packages:
```bash
pip install torch torchvision flask pillow numpy requests python-dotenv --index-url https://download.pytorch.org/whl/cpu
```
*(Using `--index-url` downloads the CPU-only version of PyTorch, which is much lighter and perfect for web application inference).*

### Step 4: Configure Environment Variables
1. Copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your Mistral API key:
   ```env
   MISTRAL_API_KEY=your_actual_mistral_api_key_here
   ```

### Step 5: Launch the Application
Start the Flask development server:
```bash
python app.py
```

### Step 6: Access the App
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌾 Supported Crops & Diseases (15 Categories)

1. **Pepper Bell:** Bacterial Spot, Healthy
2. **Potato:** Early Blight, Late Blight, Healthy
3. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
