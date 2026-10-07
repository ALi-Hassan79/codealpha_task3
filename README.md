# ✍️ AirPen — Gesture-Based Virtual Pen

AirPen is a **gesture-based virtual pen** that allows users to write and draw in the air using their hand and a webcam.

It uses **MediaPipe Hands** for real-time hand tracking and **TensorFlow.js + a local CNN model** for recognizing handwritten characters.

The goal is to provide a natural, touch-free digital writing experience for **lectures, presentations, demonstrations, and digital whiteboards**.

---

## 🚀 Features

* ✋ Real-time hand tracking using MediaPipe Hands
* 🖊️ Draw/write using the index finger
* 🧹 Erase drawings using an open-hand gesture
* 🗑️ Clear the complete canvas using a fist gesture
* 🤏 Move the drawing canvas using pinch gesture
* ✌️ Pause/stop drawing using the peace gesture
* 🎨 Adjustable pen color and brush size
* 🧈 Smooth and stable hand movement using One Euro filtering
* 🖐️ Gesture classification with multiple hand gestures
* 🤖 Automatic character/word recognition
* 🧠 Local CNN model using TensorFlow.js
* 🔤 EMNIST Balanced 47-class character recognition
* 📋 Copy recognized text to clipboard
* 💾 Save drawings as PNG images
* 🌐 Runs directly in the browser

---

## 🖐️ Gesture Controls

| Gesture         | Action        |
| --------------- | ------------- |
| ☝️ Index Finger | Draw / Write  |
| ✋ Open Hand     | Erase         |
| ✊ Fist          | Clear Canvas  |
| 🤏 Pinch        | Move Drawing  |
| ✌️ Peace        | Pause Drawing |

---

## 🤖 AI Character Recognition

AirPen includes a local **Convolutional Neural Network (CNN)** that recognizes characters written in the air.

The recognition pipeline is:

```text
Hand Movement
      ↓
MediaPipe Hands
      ↓
Canvas Drawing
      ↓
Ink Detection
      ↓
Character Segmentation
      ↓
Image Preprocessing
      ↓
28 × 28 Image
      ↓
TensorFlow.js CNN
      ↓
EMNIST Character Prediction
      ↓
Recognized Text
```

The model uses the **EMNIST Balanced dataset with 47 classes**.

The trained TensorFlow.js model is loaded locally from:

```text
models/emnist/model.json
```

The application performs recognition on the client side without sending written characters to an external API.

---

## 🧠 Technologies Used

* **JavaScript**
* **HTML5**
* **CSS3**
* **MediaPipe Hands**
* **TensorFlow.js**
* **Convolutional Neural Network (CNN)**
* **EMNIST Balanced**
* **Canvas API**
* **WebRTC / Webcam API**

---

## 📂 Project Structure

```text
AirPen/
│
├── index.html
├── app.js
│
├── models/
│   └── emnist/
│       ├── model.json
│       └── *.bin
│
└── README.md
```

---

## ⚙️ How It Works

### 1. Webcam

The browser accesses the user's webcam using the browser's media API.

### 2. Hand Detection

MediaPipe Hands detects the user's hand and tracks **21 hand landmarks** in real time.

### 3. Gesture Recognition

The application analyzes the hand landmarks to determine gestures such as:

* Point
* Open hand
* Fist
* Peace
* Pinch

The gesture classifier uses the relative positions of the fingers and palm.

### 4. Virtual Drawing

When the index finger is detected, its position is mapped onto the drawing canvas.

Movement of the finger creates a virtual pen stroke.

The application also applies smoothing to reduce hand-tracking jitter and make writing more natural.

### 5. Character Recognition

When writing stops for a short period, AirPen detects the written region, separates individual characters, preprocesses them, and sends them to the local CNN.

The CNN predicts the characters and combines them into a word.

---

## 🔤 Character Processing

Before a character is sent to the CNN, AirPen performs several preprocessing steps:

```text
Canvas
  ↓
Find Ink Region
  ↓
Character Segmentation
  ↓
Bounding Box
  ↓
Padding
  ↓
Stroke Dilation
  ↓
Resize to 28 × 28
  ↓
Tensor Conversion
  ↓
CNN Prediction
```

The application converts each character into a **28 × 28 grayscale image**, matching the expected CNN input format.

---

## 🧩 Model

The character recognition system uses:

```text
Dataset: EMNIST Balanced
Classes: 47
Input: 28 × 28 grayscale image
Model: CNN
Runtime: TensorFlow.js
```

The predicted class is mapped to the corresponding character using the EMNIST class labels.

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/ALi-Hassan79/codealpha_task3.git
```

### 2. Open the project

```bash
cd codealpha_task3
```

### 3. Start a local server

Because the project uses browser APIs and a local TensorFlow.js model, it is recommended to run it through a local web server.

For example:

```bash
python -m http.server 8000
```

### 4. Open in Browser

Go to:

```text
http://localhost:8000
```

Allow webcam access when requested.

---

## 📦 Model Requirements

The TensorFlow.js model must be available at:

```text
models/emnist/model.json
```

The corresponding model weight files must also be present in the same model directory.

AirPen loads the model using:

```javascript
tf.loadLayersModel('models/emnist/model.json');
```

---

## 🎯 Use Cases

AirPen can be used for:

* 🎓 Interactive classroom lectures
* 🧑‍🏫 Digital whiteboards
* 💻 Touch-free computer interaction
* 📝 Handwriting demonstrations
* 🔬 AI/computer-vision demonstrations
* 🖥️ Webcam-based drawing applications
* 🤖 Gesture recognition research
* 🔤 Handwritten character recognition

---

## 🔮 Future Improvements

* Improve recognition accuracy for different handwriting styles
* Add more robust word recognition
* Support numbers and additional symbols
* Improve gesture classification
* Add multilingual handwriting recognition
* Improve CNN accuracy with additional training
* Add personalized handwriting models
* Optimize recognition speed for low-end devices

---

## 👨‍💻 Author

**Ali Hassan**

GitHub:
https://github.com/ALi-Hassan79

---

## ⭐ Project

**AirPen — Gesture-Based Virtual Pen**

A computer-vision and AI project combining:

```text
Computer Vision
      +
Gesture Recognition
      +
Hand Tracking
      +
Deep Learning
      +
Character Recognition
```

Built to make digital writing more natural, interactive, and touch-free.
