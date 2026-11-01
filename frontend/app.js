const params = new URLSearchParams(location.search);
const API_URL = (
  params.get("api") ||
  import.meta.env.VITE_API_URL ||
  "https://ibm-project-5fwn.onrender.com"
).replace(/\/+$/, "");
const deviceId = `web-${crypto.randomUUID ? crypto.randomUUID() : Date.now()}`;

// ---- Sensor tuning (all overridable from the URL, no redeploy needed) ----
// ?units=ms2   send accel in m/s² (default sends g, like the UCI HAR training data)
// ?flip=0|1    force accel sign (default: flipped on iOS only)
// ?smooth=N    average the last N predictions (default 6, use 1 to disable)
// ?map=x,y,z   axis remap for accel + gyro, e.g. ?map=y,-x,z
const STEP_MS = 1000 / 50; // exactly 50 Hz
const IS_IOS = /iP(hone|ad|od)/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const ACC_SCALE = params.get("units") === "ms2" ? 1 : 1 / 9.80665;
const ACC_SIGN = params.has("flip") ? (params.get("flip") === "1" ? -1 : 1) : (IS_IOS ? -1 : 1);
const SMOOTH_N = Math.max(1, Number(params.get("smooth")) || 6);
const AXES = parseAxes(params.get("map") || "x,y,z");
function parseAxes(text) {
  const parsed = text.split(",").map((t) => {
    const m = t.trim().match(/^(-?)([xyz])$/);
    return m ? { i: "xyz".indexOf(m[2]), s: m[1] ? -1 : 1 } : null;
  });
  return parsed.length === 3 && parsed.every(Boolean) ? parsed : [{ i: 0, s: 1 }, { i: 1, s: 1 }, { i: 2, s: 1 }];
}
const remap = (v) => AXES.map(({ i, s }) => v[i] * s);

const $ = (id) => document.getElementById(id);
const sensorButton = $("sensorButton"), sensorStatus = $("sensorStatus"), sensorMessage = $("sensorMessage");
const videoInput = $("videoInput"), videoButton = $("videoButton"), videoMessage = $("videoMessage");
let selectedVideo, socket, sendTimer, nextAt = 0, collecting = false;
const samples = [], recentScores = [];

$("backendUrl").textContent = API_URL;

document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => {
  const sensorMode = button.dataset.mode === "sensor";
  document.querySelectorAll(".mode-button").forEach((tab) => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", tab === button); });
  $("sensorPanel").classList.toggle("hidden", !sensorMode);
  $("videoPanel").classList.toggle("hidden", sensorMode);
}));

function updateReading(id, value) { $(id).textContent = Number.isFinite(value) ? value.toFixed(2) : "—"; }

function onDeviceMotion(event) {
  const a = event.accelerationIncludingGravity || event.acceleration || {};
  const r = event.rotationRate || {};
  // Accel: sign-corrected and scaled (g by default). Gyro: deg/s -> rad/s.
  const acc = remap([a.x ?? 0, a.y ?? 0, a.z ?? 0].map((v) => v * ACC_SIGN * ACC_SCALE));
  const gyro = remap([r.beta ?? 0, r.gamma ?? 0, r.alpha ?? 0].map((v) => v * Math.PI / 180));
  const v = [...acc, ...gyro];
  samples.push({ t: event.timeStamp || performance.now(), v });
  if (samples.length > 120) samples.shift();
  ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"].forEach((id, i) => updateReading(id, v[i]));
}

function interpolateAt(T) {
  if (!samples.length) return null;
  if (T <= samples[0].t) return samples[0].v;
  for (let i = samples.length - 2; i >= 0; i--) {
    const p = samples[i], q = samples[i + 1];
    if (p.t <= T && T <= q.t) {
      const k = (T - p.t) / (q.t - p.t || 1);
      return p.v.map((x, j) => x + (q.v[j] - x) * k);
    }
  }
  return samples[samples.length - 1].v;
}

// Drift-corrected 50 Hz loop that sends interpolated samples with evenly spaced timestamps.
function sendLoop() {
  if (!collecting || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (performance.now() - nextAt > 500) nextAt = performance.now(); // fell far behind: resync
  const T = nextAt - 40; // stay slightly behind so two real samples bracket T
  const v = interpolateAt(T);
  if (v) socket.send(JSON.stringify({
    timestamp: (performance.timeOrigin + T) / 1000,
    total_acc_x: v[0], total_acc_y: v[1], total_acc_z: v[2],
    gyro_x: v[3], gyro_y: v[4], gyro_z: v[5],
  }));
  nextAt += STEP_MS;
  sendTimer = setTimeout(sendLoop, Math.max(0, nextAt - performance.now()));
}

async function startSensor() {
  try {
    // iOS requires this request to happen directly from a user tap.
    if (typeof DeviceMotionEvent?.requestPermission === "function") {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== "granted") throw new Error("Motion permission was not granted.");
    }
    if (!("DeviceMotionEvent" in window)) throw new Error("This browser does not provide motion sensors.");
    window.addEventListener("devicemotion", onDeviceMotion);
    const wsURL = API_URL.replace(/^http/, "ws") + `/ws/stream/${encodeURIComponent(deviceId)}`;
    socket = new WebSocket(wsURL);
    socket.onopen = () => {
      collecting = true; sensorButton.textContent = "Stop motion detection"; sensorStatus.textContent = "Live"; sensorStatus.classList.add("live");
      sensorMessage.textContent = "Collecting motion data… 128 samples needed for the first result.";
      nextAt = performance.now() + STEP_MS; sendLoop();
    };
    socket.onmessage = ({ data }) => handleSensorMessage(JSON.parse(data));
    socket.onerror = () => stopSensor("Could not connect to the backend. Is it running?");
    socket.onclose = () => { if (collecting) stopSensor("Sensor connection closed."); };
  } catch (error) { sensorMessage.textContent = error.message; }
}
function stopSensor(message = "Motion detection stopped.") {
  collecting = false; clearTimeout(sendTimer); window.removeEventListener("devicemotion", onDeviceMotion);
  samples.length = 0; recentScores.length = 0;
  if (socket && socket.readyState < WebSocket.CLOSING) socket.close(); socket = null;
  sensorButton.textContent = "Start motion detection"; sensorStatus.textContent = "Not connected"; sensorStatus.classList.remove("live"); sensorMessage.textContent = message;
}
function handleSensorMessage(data) {
  if (data.type === "buffering") { sensorMessage.textContent = `Collecting data… ${data.needed} samples until prediction.`; return; }
  if (data.type !== "prediction") return;
  // Average the last few predictions so the shown activity doesn't flicker.
  recentScores.push(data.all_scores || {});
  if (recentScores.length > SMOOTH_N) recentScores.shift();
  const avg = {};
  for (const scores of recentScores)
    for (const [name, p] of Object.entries(scores)) avg[name] = (avg[name] || 0) + p / recentScores.length;
  const top = Object.entries(avg).sort((x, y) => y[1] - x[1])[0];
  showResult(top ? { ...data, activity: top[0], confidence: top[1], all_scores: avg } : data);
}
sensorButton.addEventListener("click", () => collecting ? stopSensor() : startSensor());

function chooseVideo(file) {
  selectedVideo = file; $("videoName").textContent = file?.name || "Choose a video"; videoButton.disabled = !file;
  videoMessage.textContent = file ? `${(file.size / 1024 / 1024).toFixed(1)} MB selected.` : "Select a video clip to begin.";
}
videoInput.addEventListener("change", () => chooseVideo(videoInput.files[0]));
const dropZone = $("dropZone");
["dragenter", "dragover"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (e) => chooseVideo(e.dataTransfer.files[0]));
videoButton.addEventListener("click", async () => {
  if (!selectedVideo) return;
  videoButton.disabled = true; videoButton.textContent = "Analysing…"; videoMessage.textContent = "Uploading and analysing video…";
  try { const form = new FormData(); form.append("file", selectedVideo); const response = await fetch(`${API_URL}/video/predict`, { method: "POST", body: form }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Video prediction failed."); showResult(data); videoMessage.textContent = "Prediction complete."; }
  catch (error) { videoMessage.textContent = error.message; }
  finally { videoButton.disabled = false; videoButton.textContent = "Predict from video"; }
});

function showResult(data) {
  $("resultTitle").textContent = data.activity.replaceAll("_", " ");
  $("confidence").textContent = `Confidence ${(data.confidence * 100).toFixed(1)}%`;
  const ms = data.total_ms ?? data.inference_ms; $("timing").textContent = `${Number(ms).toFixed(0)} ms`;
  $("scoreList").innerHTML = Object.entries(data.all_scores || {}).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([name, score]) => `<div class="score"><span>${name.replaceAll("_", " ")}</span><span class="bar"><i style="width:${score * 100}%"></i></span><span>${(score * 100).toFixed(0)}%</span></div>`).join("");
}