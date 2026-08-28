# Skin Disease Classification Using Deep Learning

A comprehensive machine-learning web application for classifying dermatoscopic skin lesions into 7 diagnostic categories using the HAM10000 dataset and a Deep Learning model (MobileNetV2).

## Educational/Research Prototype
> **Disclaimer:** This system is designed for educational and research purposes only. It classifies images based on patterns learned from the HAM10000 training dataset and is not a medical diagnostic tool. Do not use its predictions to make medical decisions. Consult a qualified healthcare professional for medical evaluation.

---

## 📖 Overview
Skin cancer is a major global health issue, and early detection is crucial. This project leverages Deep Learning (Transfer Learning with MobileNetV2) to analyze dermatoscopic images and predict the likelihood of 7 different skin diseases. 

## 🎯 Problem Statement
Dermatological diagnostics often require specialized equipment and expertise. By employing computer vision, this project aims to provide a fast, preliminary analysis tool that can assist in identifying the class of a skin lesion.

## 🚀 Features
- **Modern Web UI:** A responsive, drag-and-drop interface built with HTML, CSS, and JS.
- **Deep Learning Inference:** Uses a MobileNetV2-based model trained on the HAM10000 dataset.
- **Probability Distribution:** Shows confidence scores across all 7 disease categories.
- **Data Privacy:** Local image processing without saving to external databases.

## 📊 Dataset & Classes
Trained on the **HAM10000** (Human Against Machine with 10000 training images) dataset.

| Code | Class Name |
|------|------------|
| `akiec`| Actinic Keratoses / Intraepithelial Carcinoma |
| `bcc`  | Basal Cell Carcinoma |
| `bkl`  | Benign Keratosis-like Lesions |
| `df`   | Dermatofibroma |
| `mel`  | Melanoma |
| `nv`   | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

## 🛠️ Technologies Used
- **Backend:** Python, Flask, Werkzeug
- **Machine Learning:** TensorFlow / Keras, NumPy, Pillow, Scikit-learn, Pandas
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, FontAwesome

## 🧠 Methodology
1. **Dataset Split:** Group-aware train/val/test splitting using `lesion_id` to prevent data leakage (images of the same lesion do not cross sets).
2. **Preprocessing:** Resizing to 224x224, pixel normalization [-1, 1], and heavy data augmentation for training (rotation, flips, zoom) to combat class imbalance.
3. **Model:** MobileNetV2 pre-trained on ImageNet. Stage 1 trained only the top classification head. Stage 2 unfreezes top layers for fine-tuning.
4. **Deployment:** Model is saved as `.keras` and loaded dynamically by the Flask application for real-time inference.

## 💻 Installation & How to Run

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/skin-disease-classification.git
cd skin-disease-classification

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate/Train the Model
By default, the repository does not contain the 3GB+ trained model due to size limitations. You have two options:

**Option A (For testing the UI immediately):**
Run the dummy model generator. This creates a fake model that outputs random probabilities just to test the Web App functionality.
```bash
python training/dummy_model_generator.py
```

**Option B (Train the real model):**
1. Open Google Colab.
2. Upload and run `notebooks/skin_disease_classification.ipynb`.
3. Download the generated `trained_model.keras`.
4. Place it inside the `model/` directory in this project.

### 3. Run the Web App
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure
```text
skin-disease-classification/
├── app.py                         # Main Flask application
├── config.py                      # App configuration & class mapping
├── requirements.txt               # Python dependencies
├── model/                         # Directory for trained_model.keras
├── notebooks/
│   └── skin_disease_classification.ipynb  # Colab training notebook
├── training/
│   └── dummy_model_generator.py   # Script to generate a placeholder model
├── utils/
│   ├── image_processing.py        # Image resizing and normalization
│   └── model_utils.py             # Inference logic
├── templates/
│   └── index.html                 # Main frontend UI
└── static/
    ├── css/style.css              # Custom styling
    ├── js/script.js               # Client-side logic
    └── uploads/                   # Temporary image upload storage
```

## ⚠️ Limitations & Ethical Considerations
- **Dataset Bias:** The HAM10000 dataset is predominantly composed of fair-skinned individuals. The model may underperform on darker skin tones.
- **Class Imbalance:** Despite using class weights, classes like `df` and `vasc` have far fewer samples than `nv`, affecting per-class recall.
- **Image Quality:** The model is trained on high-quality **dermatoscopic** images. Standard smartphone photos without a dermatoscope may yield unreliable results.
- **No Clinical Validation:** This prototype has not undergone clinical trials and must not be used for actual diagnosis.

## 🔮 Future Enhancements
- Integration of Grad-CAM to visualize which parts of the lesion influenced the model's decision.
- Training on a more diverse dataset (e.g., ISIC 2019/2020) to improve generalization across skin tones.
- Mobile application conversion using TensorFlow Lite.

---
**Author:** 4th-Year CSE Student  
**Year:** 2026
