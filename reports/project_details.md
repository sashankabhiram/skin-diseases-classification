# Project Report & Presentation Materials

This document contains structured content suitable for your final year project report, PPT slides, and potential Viva questions.

---

## 1. PPT Slide Content

### Slide 1: Title
**Skin Disease Classification Using Deep Learning**
*   **Subtitle:** A robust classification system using HAM10000
*   **Presenter:** [Your Name]
*   **Course:** B.Tech / BE Computer Science Engineering

### Slide 2: Problem Statement
*   Skin cancer is highly prevalent globally; early detection saves lives.
*   Dermatologist availability is limited in remote areas.
*   **Goal:** Provide an automated, preliminary analysis tool using Deep Learning to classify dermatoscopic images.

### Slide 3: Dataset (HAM10000)
*   **Dataset:** Human Against Machine with 10000 images.
*   **Classes:** 7 skin disease categories (Melanoma, Basal Cell Carcinoma, Nevi, etc.)
*   **Challenge:** Extreme class imbalance (67% of images are Melanocytic Nevi).
*   **Challenge:** Multiple images per lesion (requires careful splitting).

### Slide 4: Methodology
*   **Data Splitting:** Grouped by `lesion_id` to prevent data leakage between Train and Test sets.
*   **Preprocessing:** Resized to 224x224, pixel normalization [-1, 1].
*   **Augmentation:** Rotations, flips, and shifts to combat class imbalance.
*   **Model Architecture:** Transfer Learning using MobileNetV2.

### Slide 5: Why MobileNetV2?
*   Highly efficient and lightweight.
*   Uses Depthwise Separable Convolutions.
*   Requires fewer parameters compared to ResNet50 or VGG16, reducing training time on Colab and enabling faster CPU inference on the Flask backend.

### Slide 6: The Web Application
*   **Backend:** Python + Flask
*   **Frontend:** HTML5, CSS3, JS (Responsive UI)
*   **Flow:** User Uploads Image $\rightarrow$ Preprocessing $\rightarrow$ Model Inference $\rightarrow$ Probability Distribution Displayed.

### Slide 7: Limitations & Ethics
*   Not a replacement for a clinical dermatologist.
*   Model trained on dermatoscopic images; standard smartphone photos may reduce accuracy.
*   Dataset demographics may introduce bias against darker skin tones.

---

## 2. Expected Viva Questions & Answers

### Q1: Why did you use Transfer Learning instead of building a CNN from scratch?
**Answer:** Building a CNN from scratch requires a massive amount of data and computational power to learn basic feature extraction (like edges and curves). By using Transfer Learning with a pre-trained model like MobileNetV2 (trained on ImageNet), the model already knows how to extract features. We only need to train the final classification head for our specific 7 skin disease classes, saving time and preventing overfitting on our relatively small dataset.

### Q2: What is Data Leakage and how did you prevent it?
**Answer:** In the HAM10000 dataset, a single physical lesion might have been photographed multiple times from different angles. If we do a random 80/20 split, images of the exact same lesion could end up in both the training and testing sets. The model would just "memorize" the lesion rather than learning general disease features, leading to artificially high accuracy. I prevented this by grouping the dataset by `lesion_id` before splitting, ensuring all images of a specific lesion go strictly into either train, validation, or test.

### Q3: How did you handle the class imbalance in the HAM10000 dataset?
**Answer:** Over 67% of the dataset belongs to the 'Melanocytic Nevi' (nv) class. If ignored, the model would become biased and just predict 'nv' every time to achieve high accuracy. I addressed this using **Class Weights** during training. The loss function penalizes the model heavily when it misclassifies a minority class (like Vascular lesions) compared to a majority class. I also used data augmentation to artificially increase the variety of the minority classes.

### Q4: Why MobileNetV2 and not ResNet50?
**Answer:** While ResNet50 is very powerful, it is computationally heavy. My project required the model to run inference smoothly on a standard CPU via a Flask web backend. MobileNetV2 uses depthwise separable convolutions, making it significantly smaller and faster while maintaining comparable accuracy, making it ideal for web deployment.

### Q5: What evaluation metrics did you use? Why isn't Accuracy enough?
**Answer:** Accuracy is misleading on imbalanced datasets. If 90% of the data is Class A, a model that *always* predicts Class A gets 90% accuracy but is completely useless. Therefore, I evaluated the model using a **Confusion Matrix**, **Precision**, **Recall**, and the **Macro F1-Score**, which takes the unweighted average of F1-scores across all classes, ensuring the model performs well on minority classes too.

### Q6: Can this be used in a real hospital?
**Answer:** No. This is strictly an educational prototype. For clinical deployment, the model would need to be trained on a vastly more diverse dataset (covering all skin types and ages), undergo rigorous clinical trials, receive FDA/regulatory approval, and be calibrated to minimize false negatives (especially for fatal conditions like Melanoma).

---

## 3. Architecture Flow for Report

1. **Input:** User uploads `.jpg` image via Web UI.
2. **Server:** Flask receives file, generates unique ID, saves temporarily.
3. **Preprocessing:** PIL/NumPy resizes to `(224, 224, 3)`, converts to RGB, scales pixels to `[-1.0, 1.0]`.
4. **Inference:** `model.predict()` generates a 7-element array of softmax probabilities.
5. **Post-processing:** Server maps highest probability to class name and formats data.
6. **Output:** Frontend displays Image Preview, Predicted Class, Confidence level, and a bar chart of all probabilities.
