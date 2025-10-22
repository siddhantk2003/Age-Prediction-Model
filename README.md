# 🧠 Real-Time Age Prediction Using EfficientNet and OpenCV DNN

A deep learning project that predicts a person’s **age in real time** from a webcam feed using **EfficientNet-B0** for regression and **OpenCV DNN** for face detection.

---

## 📘 Overview

This project uses a **pre-trained EfficientNet-B0** model fine-tuned on the **UTKFace dataset** to predict the age of a person from facial images.  
The system performs the following steps in real time:

1. Detects faces from the webcam using OpenCV’s deep neural network (DNN) module.
2. Crops and preprocesses the face image.
3. Passes it through a fine-tuned EfficientNet model trained for age regression.
4. Displays the predicted age on the live video feed.

---

## ⚙️ Project Structure

```
Age-Prediction-Model/
│
├── train.py             # Script to train the EfficientNet model on UTKFace
├── utils.py             # Utility functions for preprocessing, dataset loading, etc.
├── predict_live.py      # Runs real-time age prediction using webcam feed
├── requirements.txt     # List of Python dependencies
├── models/              # Folder to store trained model weights
└── UTKFace/             # Folder to store the UTKFace Dataset
```

---

## 🧩 Features

- 🖼️ **Face Detection:** Real-time face detection using OpenCV DNN.
- ⚡ **EfficientNet-B0 Backbone:** Pre-trained model fine-tuned for regression.
- 📊 **Age Regression:** Predicts continuous age values instead of discrete classes.
- 🧠 **Transfer Learning:** Uses pretrained ImageNet weights for faster convergence.
- 💻 **Live Prediction:** Predicts age from your webcam in real-time.

---

## 🧰 Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

### Major Libraries Used
- `torch`
- `torchvision`
- `opencv-python`
- `efficientnet_pytorch`
- `numpy`
- `Pillow`

---

## 📦 Dataset: UTKFace

The [UTKFace dataset](https://susanqq.github.io/UTKFace/) contains over 20,000 facial images labeled with:
- Age
- Gender
- Ethnicity

File format:  
```
[age]_[gender]_[race]_[date&time].jpg
```

Example:  
```
25_0_2_20170116174525125.jpg
```

---

## 🚀 Training the Model

To train the model on your local machine:

```bash
python train.py
```

This will:
1. Load the UTKFace dataset.
2. Train the EfficientNet-B0 model for age prediction.
3. Save the trained weights in the `saved_models/` folder.

---

## 🎥 Running Real-Time Prediction

Once the model is trained (or using the provided pre-trained weights):

```bash
python predict_live.py
```

This script will:
- Open your webcam.
- Detect your face.
- Display the predicted age live on the video feed.

Press **‘q’** to quit.

---

## 🧠 How It Works

| Step | Description |
|------|--------------|
| **1. Face Detection** | OpenCV DNN locates faces using a Caffe-based model. |
| **2. Preprocessing** | Image is cropped, resized to 224x224, and normalized. |
| **3. Age Prediction** | EfficientNet-B0 predicts the person’s age as a continuous value. |
| **4. Output Display** | The predicted age is displayed in real time on the webcam frame. |

---

## 🧪 Results

- Model trained on ~20,000 UTKFace images.
- Achieved a **Mean Absolute Error (MAE)** of approximately **±3 years** on validation data.
- Real-time performance: ~25 FPS on GPU / ~10 FPS on CPU.

---

## 💡 Future Improvements

- Add gender and ethnicity prediction.
- Improve accuracy using EfficientNet-B3 or Swin Transformer.
- Build a GUI dashboard for visualization.
- Deploy as a web application using Streamlit or Flask.



---

## 🪪 License

This project is licensed under the **MIT License**.  
You are free to use and modify it for educational or research purposes.

---

⭐ *If you like this project, don’t forget to give it a star on GitHub!*