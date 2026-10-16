/* ============ Live Emotion Detection - Frontend ============ */

const EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
const EMOTION_UI_COLORS = {
  Angry: "#e74c3c",
  Disgust: "#8e6f3e",
  Fear: "#9b59b6",
  Happy: "#2ecc71",
  Neutral: "#95a5a6",
  Sad: "#3498db",
  Surprise: "#f1c40f",
};

/* ---------------- Tabs ---------------- */
const tabs = document.querySelectorAll(".tab");
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(`tab-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

/* ---------------- Helpers ---------------- */
function barRows(probabilities) {
  return EMOTIONS.map((name) => {
    const pct = Math.round((probabilities[name] || 0) * 100);
    const color = EMOTION_UI_COLORS[name];
    return `
      <div class="bar-row">
        <span class="label">${name}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="pct">${pct}%</span>
      </div>`;
  }).join("");
}

function faceCard(face, index) {
  return `
    <div class="face-card">
      <div class="face-head">
        <span class="face-emotion" style="color:${EMOTION_UI_COLORS[face.emotion]}">
          ${face.emotion}
        </span>
        <span class="face-conf">Face ${index + 1} · ${(face.confidence * 100).toFixed(0)}%</span>
      </div>
      ${barRows(face.probabilities)}
    </div>`;
}

/* ---------------- Live webcam ---------------- */
const streamImg = document.getElementById("stream");
const placeholder = document.getElementById("stream-placeholder");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const liveStatus = document.getElementById("live-status");
const liveFaces = document.getElementById("live-faces");

let pollTimer = null;
let lastFacesKey = null;

async function startLive() {
  btnStart.disabled = true;
  liveStatus.textContent = "Starting camera...";
  try {
    const res = await fetch("/api/live/start", { method: "POST" });
    const data = await res.json();
    if (!data.started) {
      liveStatus.textContent = data.error || "Failed to start camera.";
      btnStart.disabled = false;
      return;
    }
    // Cache-buster so the <img> reconnects to a fresh stream
    streamImg.src = "/video_feed?t=" + Date.now();
    streamImg.hidden = false;
    placeholder.style.display = "none";
    btnStop.disabled = false;
    liveStatus.textContent = "Live";
    lastFacesKey = null;
    pollTimer = setInterval(pollPredictions, 700);
  } catch (err) {
    liveStatus.textContent = "Server unreachable: " + err;
    btnStart.disabled = false;
  }
}

async function stopLive() {
  clearInterval(pollTimer);
  pollTimer = null;
  await fetch("/api/live/stop", { method: "POST" });
  streamImg.src = "";
  streamImg.hidden = true;
  placeholder.style.display = "block";
  liveFaces.innerHTML = '<p class="placeholder small">No face detected yet.</p>';
  liveStatus.textContent = "Idle";
  btnStop.disabled = true;
  btnStart.disabled = false;
}

async function pollPredictions() {
  try {
    const res = await fetch("/api/predictions");
    const data = await res.json();
    if (data.error) {
      liveStatus.textContent = data.error;
      return;
    }
    liveStatus.textContent = `Live · ${data.fps} FPS`;

    // Only touch the DOM when the detections actually changed.
    // Re-rendering the bar charts every poll causes visible jank.
    const key = JSON.stringify(data.faces.map((f) => [f.emotion, Math.round(f.confidence * 100)]));
    if (key === lastFacesKey) return;
    lastFacesKey = key;

    if (data.faces.length === 0) {
      liveFaces.innerHTML = '<p class="placeholder small">No face detected yet.</p>';
    } else {
      liveFaces.innerHTML = data.faces.map(faceCard).join("");
    }
  } catch (_) {
    /* transient polling error, ignore */
  }
}

btnStart.addEventListener("click", startLive);
btnStop.addEventListener("click", stopLive);

// Stop the camera automatically when leaving the page
window.addEventListener("beforeunload", () => {
  if (!btnStop.disabled) navigator.sendBeacon("/api/live/stop");
});

/* ---------------- Image upload ---------------- */
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const btnAnalyze = document.getElementById("btn-analyze");
const uploadForm = document.getElementById("upload-form");
const uploadError = document.getElementById("upload-error");
const uploadResult = document.getElementById("upload-result");

let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please choose an image file (JPG / PNG).");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  btnAnalyze.disabled = false;
  hideError();
}

function showError(msg) {
  uploadError.textContent = msg;
  uploadError.classList.remove("hidden");
}
function hideError() {
  uploadError.classList.add("hidden");
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  btnAnalyze.disabled = true;
  btnAnalyze.textContent = "Analyzing...";
  hideError();

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      uploadResult.innerHTML =
        '<p class="placeholder small">Upload an image to see the detected emotion.</p>';
      return;
    }

    uploadResult.innerHTML =
      `<img src="data:image/jpeg;base64,${data.image}" alt="Annotated result" />` +
      data.faces.map(faceCard).join("");
  } catch (err) {
    showError("Server error: " + err);
  } finally {
    btnAnalyze.textContent = "Analyze Emotion";
    btnAnalyze.disabled = false;
  }
});
