import cv2
import numpy as np
from fer import FER
from tensorflow.keras.models import load_model

# --- Load Mini-XCEPTION model ---
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Emotion labels
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

# --- Face detectors ---
fer_detector = FER(mtcnn=True)  # MTCNN (better but slower)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Webcam ---
cap = cv2.VideoCapture(0)
print("Hybrid Emotion Detection with fallback started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = fer_detector.detect_emotions(frame)  # detect with MTCNN

    # --- If MTCNN fails, fallback to Haar ---
    if len(faces) == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = haar_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Convert Haar format to FER-like format
        faces = [{"box": (x, y, w, h)} for (x, y, w, h) in haar_faces]

    for face in faces:
        (x, y, w, h) = face["box"]
        # Clamp values to avoid errors
        x, y = max(0, x), max(0, y)
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size == 0:  # skip empty crops
            continue

        # Preprocess for Mini-XCEPTION
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.equalizeHist(gray_face)  # enhance contrast
        final_face = cv2.resize(gray_face, (64, 64))
        final_face = final_face / 255.0
        final_face = np.expand_dims(final_face, axis=-1)  # add channel
        final_face = np.expand_dims(final_face, axis=0)   # add batch

        # Predict emotion
        predictions = emotion_model.predict(final_face, verbose=0)
        emotion_index = np.argmax(predictions)
        dominant_emotion = EMOTION_LABELS[emotion_index]
        emotion_score = predictions[0][emotion_index]

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{dominant_emotion}: {emotion_score:.2f}"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Hybrid Emotion Detection (MTCNN + Haar Fallback)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
