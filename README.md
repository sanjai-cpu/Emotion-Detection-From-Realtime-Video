# 🎭 Hybrid Emotion Detection from Real-Time Video (MTCNN + Haar Cascade)

This project detects **human emotions in real-time** using a webcam feed.  
It combines **deep learning-based emotion recognition** with **hybrid face detection** —  
leveraging **MTCNN (Multi-task Cascaded Convolutional Networks)** for high accuracy and **Haar Cascade** as a fast fallback when MTCNN fails.

---

## 🧠 Features
✅ Real-time webcam emotion detection  
✅ Hybrid face detection (MTCNN + Haar Cascade fallback)  
✅ Uses **Mini-XCEPTION** model trained on **FER2013** dataset  
✅ Displays emotion label and confidence score on screen  
✅ Works even in low-light or occluded face conditions  

---

## 🧩 Tech Stack
- **Python 3.8+**
- **OpenCV** (video and face detection)
- **FER (Facial Expression Recognition)** library (MTCNN-based detection)
- **TensorFlow / Keras** (for Mini-XCEPTION model)
- **NumPy** (data preprocessing)

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sanjai-cpu/Emotion-Detection-From-Realtime-Video.git
cd Emotion-Detection-From-Realtime-Video
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, create one with:

bash
Copy code
opencv-python
fer
tensorflow
numpy
3️⃣ Download the model file
Place the pre-trained Mini-XCEPTION model file in the project folder:

Copy code
fer2013_mini_XCEPTION.102-0.66.hdf5
You can download it from:
👉 Mini-XCEPTION Model (FER2013)

🚀 Usage
Run the program:

bash
Copy code
python emotion_detection.py
Once the webcam window opens:

The system detects your face and predicts your dominant emotion.

Press q to quit the application.

📦 Project Structure
css
Copy code
📁 Emotion-Detection-From-Realtime-Video/
├── emotion_detection.py          # Main program file
├── fer2013_mini_XCEPTION.102-0.66.hdf5   # Pre-trained model
├── requirements.txt
└── README.md
😄 Supported Emotions
Label	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Sad
5	Surprise
6	Neutral

🧠 How It Works
Frame Capture: Grabs each frame from webcam feed.

Face Detection:

Tries MTCNN (accurate).

Falls back to Haar Cascade if MTCNN fails.

Preprocessing: Converts the face region to grayscale, resizes to 64×64, and normalizes.

Emotion Prediction: Uses the Mini-XCEPTION model to predict emotion probabilities.

Display: Draws a rectangle and emotion label on the detected face.

🧩 Example Output
csharp
Copy code
Hybrid Emotion Detection with fallback started... Press 'q' to quit.

💡 Future Improvements
Add GPU acceleration for faster inference (TensorRT or ONNX)

Implement multi-face tracking

Integrate with a dashboard for emotion analytics

