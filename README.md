# 🎭 Live Emotion Detection

## 📌 Project Overview

This project detects human emotions in real-time using a webcam.
A Convolutional Neural Network (CNN) model is trained on facial expression images to classify emotions like **angry, happy, sad, fear, surprise, neutral, and disgust**.

---
## 📥 Dataset
This project utilizes the **FER-2013 (Facial Expression Recognition)** dataset, one of the most widely used datasets for emotion AI.

* **Format:** Grayscale facial images (48x48 pixels).
* **Total Classes:** 7 emotion categories.
    * 0: Angry
    * 1: Disgust
    * 2: Fear
    * 3: Happy
    * 4: Sad
    * 5: Surprise
    * 6: Neutral
* **Data Split:** Images are organized into `train` and `test` directories.

🔗 **Download Link:** [Kaggle - FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

> **Note:** A Kaggle account is required to download the files. Once downloaded, extract the ZIP file into the `data/` folder.

---
## ⚙️ Environment Setup

* **Operating System:** Windows 10 / 11
* **Python Version:** 3.10.5

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
live_emotion_detection/
│
├── data/                         # Dataset (train/test images)
├── templates/
│   └── index.html               # Web app UI
├── static/
│   ├── style.css                # Web app styling
│   └── script.js                # Web app frontend logic
│
├── load_dataset.py              # Load dataset using ImageDataGenerator
├── check_imbalance.py           # Analyze class distribution
├── compute_weights.py           # Compute class weights
├── data_augmentation.py         # Apply data augmentation
├── build_model.py               # Build CNN model
├── train_model.py               # Train model
├── emotion_engine.py            # Shared face + emotion detection logic
├── app.py                       # Flask web application
├── realtime_detection.py        # Live emotion detection (desktop webcam)
│
├── best_emotion_model.keras     # Saved best model
├── requirements.txt             # Required packages
```

---

## 🔄 Process Flow (Execution Order)

Follow this order step-by-step:

---

### 1️⃣ Load Dataset

📄 `load_dataset.py`

* Loads images from dataset folders
* Converts images to required format (48x48 grayscale)
* Splits into training and validation

---

### 2️⃣ Check Class Imbalance

📄 `check_imbalance.py`

* Counts number of images per emotion
* Displays imbalance using bar chart
* Helps identify minority classes (like *disgust*)

---

### 3️⃣ Compute Class Weights

📄 `compute_weights.py`

* Calculates weights for each class
* Assigns higher importance to minority classes
* Helps reduce model bias

---

### 4️⃣ Data Augmentation

📄 `data_augmentation.py`

* Applies transformations:

  * Rotation
  * Shift
  * Zoom
  * Flip
* Improves model generalization
* Prevents overfitting

---

### 5️⃣ Build CNN Model

📄 `build_model.py`

* Creates deep CNN architecture:

  * Conv layers (32 → 64 → 128 → 256)
  * Batch Normalization
  * Dropout layers
* Designed for better feature extraction

---

### 6️⃣ Train Model

📄 `train_model.py`

* Trains model using:

  * Augmented data
  * Class weights
* Uses:

  * EarlyStopping
  * ModelCheckpoint
* Saves best model

---

### 7️⃣ Real-Time Emotion Detection

📄 `realtime_detection.py`

* Uses webcam via OpenCV
* Detects face using Haarcascade
* Predicts emotion in real-time
* Displays emotion label on screen

---

### 8️⃣ Web Application 🌐

📄 `app.py` (Flask)

Run:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Two modes:

* **📹 Live Webcam** – start/stop the camera and see real-time emotion
  detection streamed in the browser, with live confidence bars for every face.
* **🖼️ Upload Image** – drop any photo and get the detected emotion(s) with
  per-emotion probabilities and an annotated result image.

The shared detection logic lives in `emotion_engine.py`, so the web app and
scripts always use the same model and preprocessing. To use a different model
file, set the `EMOTION_MODEL_PATH` environment variable before starting.

---

## 🎯 Model Performance

* Validation Accuracy: **~56%**
* Dataset: FER-2013 (challenging dataset)

---

## ⚠️ Limitations

* Dataset imbalance (few samples for *disgust*)
* Similar emotions may confuse model
* Accuracy depends on lighting and face clarity

---

## 🚀 Future Improvements

* Use Transfer Learning (MobileNet / ResNet)
* Add more training data
* Improve face detection accuracy

---

## 🏁 Conclusion

This project demonstrates a complete pipeline:

* Data preprocessing
* Model training
* Real-time deployment

It is suitable for learning and practical AI applications.

---
