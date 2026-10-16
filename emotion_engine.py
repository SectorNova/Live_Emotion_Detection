"""
emotion_engine.py
Shared "brain" for emotion detection: face detection (Haarcascade) +
CNN emotion prediction. Used by the Flask web app (app.py).
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model saved by train_model.py (override with EMOTION_MODEL_PATH env var)
MODEL_PATH = os.environ.get(
    "EMOTION_MODEL_PATH",
    os.path.join(BASE_DIR, "best_emotion_model.keras"),
)

# Alphabetical order matches the training folder order (ImageDataGenerator)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# One color per emotion (BGR, for OpenCV drawing)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 128),
    "Fear": (255, 0, 255),
    "Happy": (0, 200, 0),
    "Neutral": (128, 128, 128),
    "Sad": (255, 100, 0),
    "Surprise": (0, 255, 255),
}

IMG_SIZE = (48, 48)
DET_WIDTH = 320          # face detection runs on a small proxy image (speed!)
STREAM_JPEG_QUALITY = 70


class EmotionEngine:
    """Loads the trained CNN once and provides detection on images/frames."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model: {os.path.basename(model_path)} ...")
        # compile=False -> only weights/architecture are needed for inference
        self.model = load_model(model_path, compile=False)

        # Fast inference path: a compiled tf.function is much cheaper per call
        # than model.predict(), especially for the tiny 48x48 batch.
        try:
            infer = tf.function(lambda x: self.model(x, training=False))
            infer.get_concrete_function(
                tf.TensorSpec([None, IMG_SIZE[0], IMG_SIZE[1], 1], tf.float32)
            )
            self._run = lambda arr: np.asarray(infer(arr))
        except Exception:
            self._run = lambda arr: self.model.predict(arr, verbose=0)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from: {cascade_path}")
        print("Emotion engine ready.")

    # ------------------------------------------------------------- inference
    def detect_faces(self, bgr_image, min_neighbors=5):
        """
        Detect all faces in a BGR image and predict an emotion for each.

        Speed tricks:
          * Haar runs on a DET_WIDTH-wide proxy (one predict call is ~4x the
            cost per pixel at full size); boxes are mapped back to full coords.
          * All faces are classified in ONE batched model call.

        Returns a list of dicts:
            {box: (x, y, w, h), emotion, confidence,
             probabilities: {label: float, ...}}
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Downscale only for reasonably large frames (tiny crops stay as-is)
        if w >= 480:
            s = DET_WIDTH / float(w)
            gray_det = cv2.resize(
                gray, (DET_WIDTH, max(1, int(h * s))), interpolation=cv2.INTER_AREA
            )
        else:
            s = 1.0
            gray_det = gray

        faces = self.face_cascade.detectMultiScale(
            gray_det,
            scaleFactor=1.3,
            minNeighbors=min_neighbors,
            minSize=(max(16, int(30 * s)), max(16, int(30 * s))),
        )
        if len(faces) == 0:
            return []

        # One batched CNN call for every face in this frame
        rois, kept = [], []
        for (x, y, w_, h_) in faces:
            if s == 1.0:
                roi = gray[y : y + h_, x : x + w_]
            else:
                roi = gray[int(y / s) : int((y + h_) / s), int(x / s) : int((x + w_) / s)]
            if roi.size == 0:
                continue
            rois.append(cv2.resize(roi, IMG_SIZE))
            kept.append((x, y, w_, h_))
        if not rois:
            return []
        rois = np.stack(rois).astype(np.float32) / 255.0
        probs_all = self._run(rois)

        results = []
        for (x, y, w_, h_), probs in zip(kept, probs_all):
            idx = int(np.argmax(probs))
            results.append(
                {
                    "box": (int(x / s), int(y / s), int(w_ / s), int(h_ / s)),
                    "emotion": EMOTION_LABELS[idx],
                    "confidence": float(probs[idx]),
                    "probabilities": {
                        label: float(p) for label, p in zip(EMOTION_LABELS, probs)
                    },
                }
            )
        return results

    # ------------------------------------------------------------- annotate
    def annotate(self, bgr_image, results, show_confidence=True):
        """Draw bounding boxes + emotion labels on a copy of the image."""
        out = bgr_image.copy()
        for r in results:
            x, y, w, h = r["box"]
            color = EMOTION_COLORS.get(r["emotion"], (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            label = r["emotion"]
            if show_confidence:
                label += f" {r['confidence'] * 100:.0f}%"

            # Filled tag background so text stays readable on any frame
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            tag_y = max(y - 6, th + 10)
            cv2.rectangle(out, (x, tag_y - th - 8), (x + tw + 8, tag_y + 4), color, -1)
            cv2.putText(
                out,
                label,
                (x + 4, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out
