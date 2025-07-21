# 🩺 Chest X-Ray Classifier: COVID-19, Pneumonia & Normal Detection

This project is a deep learning-based classification system built with Convolutional Neural Networks (CNNs) to detect **COVID-19**, **Pneumonia**, and **Normal** cases from **Chest X-ray images**.

![Chest X-ray samples](https://storage.googleapis.com/kaggle-datasets-images/1573326/2553342/8e25c1050dcf0e31459821cd31e0cddc/dataset-card.jpg)

---

## 📁 Dataset

Dataset used: [Chest X-ray COVID19 Pneumonia](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)  
It contains chest X-ray images in the following structure:

chest_xray/
├── train/
│ ├── COVID19/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── COVID19/
├── NORMAL/
└── PNEUMONIA/


- **Total Images**: ~6,000
- **Classes**: `COVID19`, `NORMAL`, `PNEUMONIA`

---

## 🧠 Model Architecture

Built a **CNN using TensorFlow (Keras)** with the following architecture:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 output classes
])

              precision    recall  f1-score   support

     COVID19       0.96      0.91      0.94       116
      NORMAL       0.82      0.95      0.88       317
   PNEUMONIA       0.97      0.92      0.95       855
