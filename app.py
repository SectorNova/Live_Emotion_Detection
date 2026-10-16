"""
app.py
Flask web application for Live Emotion Detection.

Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.

Features:
  - Live webcam emotion detection (MJPEG stream in the browser)
  - Emotion detection on an uploaded image
"""

import base64
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from emotion_engine import STREAM_JPEG_QUALITY, EMOTION_LABELS, EmotionEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

print("Starting server, loading model...")
engine = EmotionEngine()


# --------------------------------------------------------------------------
# Webcam worker: reads frames, runs detection, keeps latest JPEG + results
# --------------------------------------------------------------------------
class CameraStream:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._frame_jpeg = None   # latest annotated frame (bytes)
        self._frame_id = 0        # increments only when a new frame exists
        self._results = []        # latest detection results
        self._fps = 0.0
        self._error = None

    @property
    def active(self):
        return self._running

    def start(self):
        with self._lock:
            if self._running:
                return True, self._error
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                return False, "Could not open webcam. Is another app using it?"
            # Ask for a small capture resolution: reading 720p+ webcams and
            # processing them is the main source of lag on CPU laptops.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lowest-latency frame
            except Exception:
                pass
            self._cap = cap
            self._error = None
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True, None

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if getattr(self, "_cap", None):
            self._cap.release()
            self._cap = None

    def _loop(self):
        last = time.time()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._error = "Lost webcam connection."
                time.sleep(0.2)
                continue

            results = engine.detect_faces(frame)
            annotated = engine.annotate(frame, results)

            # Downscale a bit so the browser stream stays smooth on CPU
            h, w = annotated.shape[:2]
            if w > 640:
                scaled = 640 / w
                annotated = cv2.resize(
                    annotated, (640, int(h * scaled)), interpolation=cv2.INTER_AREA
                )

            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )
            if not ok:
                continue

            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now

            with self._lock:
                self._frame_jpeg = buf.tobytes()
                self._frame_id += 1
                self._results = results
                self._fps = fps

    def snapshot(self):
        with self._lock:
            return self._frame_jpeg, self._frame_id, list(self._results), self._fps, self._error


camera = CameraStream()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", emotions=EMOTION_LABELS)


@app.route("/api/live/start", methods=["POST"])
def live_start():
    ok, err = camera.start()
    return jsonify({"started": ok, "error": err}), (200 if ok else 503)


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    camera.stop()
    return jsonify({"stopped": True})


@app.route("/video_feed")
def video_feed():
    """MJPEG stream consumed by <img> tag. Sends each frame exactly once."""

    def generate():
        last_id = -1
        while True:
            jpeg, frame_id, _, _, _ = camera.snapshot()
            if not camera.active and frame_id == last_id:
                break
            if jpeg and frame_id != last_id:
                last_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.02)

    return Flask.response_class(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/predictions")
def api_predictions():
    """Latest live-mode results, polled by the frontend for the chart."""
    _, _, results, fps, error = camera.snapshot()
    return jsonify(
        {
            "active": camera.active,
            "fps": round(fps, 1),
            "error": error,
            "faces": results,
        }
    )


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Detect emotions in an uploaded image."""
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not read that file as an image."}), 400

    # Keep processing fast on CPU-laptops
    h, w = image.shape[:2]
    if max(h, w) > 900:
        scaled = 900 / max(h, w)
        image = cv2.resize(
            image, (int(w * scaled), int(h * scaled)), interpolation=cv2.INTER_AREA
        )
    elif max(h, w) < 160:
        # Tiny crops (e.g. FER-2013 48x48 images): upscale so Haar can see the face
        image = cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    results = engine.detect_faces(image, min_neighbors=4)
    if not results:
        return jsonify({"error": "No face found in the image. Try a clearer, front-facing photo."}), 422

    annotated = engine.annotate(image, results)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return jsonify({"image": image_b64, "faces": results, "face_count": len(results)})


if __name__ == "__main__":
    # threaded=True so the browser can show video + poll JSON at the same time
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
