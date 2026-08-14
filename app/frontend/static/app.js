const fileInput = document.getElementById("file-input");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const eraseBtn = document.getElementById("erase-btn");
const statusEl = document.getElementById("status");
const resultImg = document.getElementById("result");
const downloadLink = document.getElementById("download-link");

let originalFile = null;
let originalImage = null;
let clickPoint = null; // координаты в пикселях оригинального изображения
let scale = 1;

const MAX_CANVAS_DIM = 500;

function setStatus(text) {
  statusEl.textContent = text;
}

async function readErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  originalFile = file;
  const image = new Image();
  image.onload = () => {
    originalImage = image;
    scale = Math.min(1, MAX_CANVAS_DIM / Math.max(image.width, image.height));
    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    clickPoint = null;
    eraseBtn.disabled = true;
    resultImg.hidden = true;
    downloadLink.hidden = true;
    setStatus("Кликни по объекту, который нужно убрать");
  };
  image.src = URL.createObjectURL(file);
});

canvas.addEventListener("click", (event) => {
  if (!originalImage) return;

  const rect = canvas.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (event.clientY - rect.top) * (canvas.height / rect.height);
  clickPoint = { x: Math.round(canvasX / scale), y: Math.round(canvasY / scale) };

  ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.arc(canvasX, canvasY, 6, 0, Math.PI * 2);
  ctx.fillStyle = "red";
  ctx.fill();

  eraseBtn.disabled = false;
  setStatus(`Точка выбрана (${clickPoint.x}, ${clickPoint.y}). Нажми «Стереть».`);
});

eraseBtn.addEventListener("click", async () => {
  if (!originalFile || !clickPoint) return;

  eraseBtn.disabled = true;
  resultImg.hidden = true;
  downloadLink.hidden = true;

  try {
    setStatus("Ищу объект...");
    const segmentForm = new FormData();
    segmentForm.append("file", originalFile);
    segmentForm.append("x", clickPoint.x);
    segmentForm.append("y", clickPoint.y);

    const segmentResponse = await fetch("/segment", { method: "POST", body: segmentForm });
    if (!segmentResponse.ok) {
      throw new Error(await readErrorDetail(segmentResponse));
    }
    const maskBlob = await segmentResponse.blob();

    setStatus("Закрашиваю...");
    const inpaintForm = new FormData();
    inpaintForm.append("file", originalFile);
    inpaintForm.append("mask", maskBlob, "mask.png");

    const inpaintResponse = await fetch("/inpaint", { method: "POST", body: inpaintForm });
    if (!inpaintResponse.ok) {
      throw new Error(await readErrorDetail(inpaintResponse));
    }
    const resultBlob = await inpaintResponse.blob();
    const resultUrl = URL.createObjectURL(resultBlob);

    resultImg.src = resultUrl;
    resultImg.hidden = false;
    downloadLink.href = resultUrl;
    downloadLink.hidden = false;
    setStatus("Готово!");
  } catch (error) {
    setStatus(`Ошибка: ${error.message}`);
  } finally {
    eraseBtn.disabled = false;
  }
});
