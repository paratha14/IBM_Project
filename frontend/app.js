// Change this to your machine's LAN IP when opening the page on a phone.
const API_URL = new URLSearchParams(location.search).get("api") || "http://localhost:8000";
const SAMPLE_RATE_MS = 20;
const deviceId = `web-${crypto.randomUUID ? crypto.randomUUID() : Date.now()}`;

const $ = (id) => document.getElementById(id);
const sensorButton = $("sensorButton"), sensorStatus = $("sensorStatus"), sensorMessage = $("sensorMessage");
const videoInput = $("videoInput"), videoButton = $("videoButton"), videoMessage = $("videoMessage");
let selectedVideo, socket, latestMotion, sendTimer, collecting = false;

$("backendUrl").textContent = API_URL;

document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => {
  const sensorMode = button.dataset.mode === "sensor";
  document.querySelectorAll(".mode-button").forEach((tab) => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", tab === button); });
  $("sensorPanel").classList.toggle("hidden", !sensorMode);
  $("videoPanel").classList.toggle("hidden", sensorMode);
}));

function updateReading(id, value) { $(id).textContent = Number.isFinite(value) ? value.toFixed(2) : "—"; }
function onDeviceMotion(event) {
  // accelerationIncludingGravity gives the backend total_acc fields in m/s².
  const acceleration = event.accelerationIncludingGravity || event.acceleration;
  const rotation = event.rotationRate || {};
  latestMotion = {
    timestamp: Date.now() / 1000,
    total_acc_x: acceleration?.x ?? 0, total_acc_y: acceleration?.y ?? 0, total_acc_z: acceleration?.z ?? 0,
    // DeviceMotion rotationRate is degrees/s; convert it to rad/s for the trained model.
    gyro_x: (rotation.beta ?? 0) * Math.PI / 180, gyro_y: (rotation.gamma ?? 0) * Math.PI / 180, gyro_z: (rotation.alpha ?? 0) * Math.PI / 180,
  };
  updateReading("accelX", latestMotion.total_acc_x); updateReading("accelY", latestMotion.total_acc_y); updateReading("accelZ", latestMotion.total_acc_z);
  updateReading("gyroX", latestMotion.gyro_x); updateReading("gyroY", latestMotion.gyro_y); updateReading("gyroZ", latestMotion.gyro_z);
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
      sendTimer = setInterval(() => { if (socket.readyState === WebSocket.OPEN && latestMotion) socket.send(JSON.stringify(latestMotion)); }, SAMPLE_RATE_MS);
    };
    socket.onmessage = ({ data }) => handleSensorMessage(JSON.parse(data));
    socket.onerror = () => stopSensor("Could not connect to the backend. Is it running?");
    socket.onclose = () => { if (collecting) stopSensor("Sensor connection closed."); };
  } catch (error) { sensorMessage.textContent = error.message; }
}
function stopSensor(message = "Motion detection stopped.") {
  collecting = false; clearInterval(sendTimer); window.removeEventListener("devicemotion", onDeviceMotion);
  if (socket && socket.readyState < WebSocket.CLOSING) socket.close(); socket = null;
  sensorButton.textContent = "Start motion detection"; sensorStatus.textContent = "Not connected"; sensorStatus.classList.remove("live"); sensorMessage.textContent = message;
}
function handleSensorMessage(data) {
  if (data.type === "buffering") { sensorMessage.textContent = `Collecting data… ${data.needed} samples until prediction.`; return; }
  if (data.type === "prediction") showResult(data);
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
