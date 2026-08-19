const wizardCard = document.getElementById("wizard-card");
const panels = {
  upload: document.getElementById("panel-upload"),
  select: document.getElementById("panel-select"),
  result: document.getElementById("panel-result"),
};

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileNameLabel = document.getElementById("file-name-label");
const canvasWrap = document.getElementById("canvas-wrap");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const processingText = document.getElementById("processing-text");
const changePhotoBtn = document.getElementById("change-photo-btn");
const eraseBtn = document.getElementById("erase-btn");
const eraseBtnLabel = document.getElementById("erase-btn-label");
const banner = document.getElementById("banner");
const bannerIcon = document.getElementById("banner-icon");
const bannerText = document.getElementById("banner-text");
const resultSkeleton = document.getElementById("result-skeleton");
const resultImg = document.getElementById("result");
const downloadLink = document.getElementById("download-link");
const startOverBtn = document.getElementById("start-over-btn");
const stepEls = {
  1: document.getElementById("step-1"),
  2: document.getElementById("step-2"),
  3: document.getElementById("step-3"),
};

let originalFile = null;
let originalImage = null;
let clickPoint = null; // координаты в пикселях оригинального изображения
let scale = 1;

const MAX_CANVAS_DIM = 700;

const BANNER_ICONS = {
  info: "ℹ️",
  success: "✅",
  warning: "⚠️",
  error: "⛔",
};

function showBanner(kind, text) {
  banner.className = `banner visible banner-${kind}`;
  bannerIcon.textContent = BANNER_ICONS[kind];
  bannerText.textContent = text;
}

function hideBanner() {
  banner.className = "banner";
}

// Wizard: на экране только панель текущего шага — предыдущий шаг (например,
// дропзона) убирается из разметки, а не просто визуально теряется среди
// остального контента. Переход всегда сопровождается прокруткой к началу
// карточки и fade-in новой панели, чтобы смена шага была явно заметна.
function goToStep(step, { processing = false } = {}) {
  const panelByStep = { 1: "upload", 2: "select", 3: "result" };
  Object.entries(panels).forEach(([name, el]) => {
    el.classList.toggle("visible", name === panelByStep[step]);
  });
  Object.entries(stepEls).forEach(([num, el]) => {
    const n = Number(num);
    el.classList.toggle("active", n === step && !processing);
    el.classList.toggle("processing", n === step && processing);
    el.classList.toggle("done", n < step);
  });
  wizardCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Помимо текста и спиннера в кнопке — затемняем сам канвас с крупным спиннером
// и подписью этапа поверх фото, чтобы процесс был заметен даже боковым зрением,
// а не только в мелком тексте кнопки.
function setProcessing(isProcessing, label) {
  eraseBtn.disabled = isProcessing;
  eraseBtn.classList.toggle("loading", isProcessing);
  eraseBtnLabel.textContent = isProcessing ? label : "Стереть";
  canvasWrap.classList.toggle("processing", isProcessing);
  processingText.textContent = label;
  changePhotoBtn.disabled = isProcessing;
  if (isProcessing) {
    goToStep(2, { processing: true });
  }
}

// Классифицируем HTTP-статус в вид баннера и человекочитаемый текст,
// чтобы 413/422/500 и т.п. не выглядели как одна и та же "ошибка".
async function describeErrorResponse(response) {
  let detail = null;
  try {
    const data = await response.json();
    detail = data.detail || null;
  } catch {
    detail = null;
  }

  const status = response.status;
  if (status === 413) {
    return detail || "Файл слишком большой. Попробуйте изображение меньшего размера.";
  }
  if (status === 400 || status === 422) {
    return detail || "Не удалось обработать файл — проверьте, что это корректное изображение.";
  }
  if (status === 403) {
    return detail || "Запрос отклонён сервером.";
  }
  if (status >= 500) {
    return "Сервис временно недоступен. Попробуйте ещё раз чуть позже.";
  }
  return detail || response.statusText || "Не удалось выполнить запрос.";
}

function resetToUpload() {
  originalFile = null;
  originalImage = null;
  clickPoint = null;
  fileInput.value = "";
  eraseBtn.disabled = true;
  resultSkeleton.classList.remove("visible");
  resultImg.hidden = true;
  downloadLink.hidden = true;
  startOverBtn.hidden = true;
  panels.result.classList.remove("just-arrived");
  hideBanner();
  goToStep(1);
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showBanner("error", "Выберите файл изображения (PNG, JPG и т.п.)");
    return;
  }

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
    fileNameLabel.textContent = file.name;
    hideBanner();
    goToStep(2);

    // Явный, заметный отклик на сам факт загрузки: фото не просто "тихо"
    // появляется — рамка вспыхивает акцентным цветом и гаснет.
    canvasWrap.classList.remove("just-loaded");
    // eslint-disable-next-line no-unused-expressions
    void canvasWrap.offsetWidth; // форсируем reflow, чтобы анимация перезапустилась
    canvasWrap.classList.add("just-loaded");
    showBanner("info", `Загружено: ${file.name}. Кликните по объекту, который нужно убрать.`);
  };
  image.onerror = () => {
    showBanner("error", "Не удалось прочитать это изображение в браузере.");
  };
  image.src = URL.createObjectURL(file);
}

fileInput.addEventListener("change", (event) => {
  loadFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) loadFile(file);
});

changePhotoBtn.addEventListener("click", resetToUpload);
startOverBtn.addEventListener("click", resetToUpload);

canvas.addEventListener("click", (event) => {
  if (!originalImage) return;

  const rect = canvas.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (event.clientY - rect.top) * (canvas.height / rect.height);
  clickPoint = { x: Math.round(canvasX / scale), y: Math.round(canvasY / scale) };

  ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.arc(canvasX, canvasY, 7, 0, Math.PI * 2);
  ctx.fillStyle = "#e33";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#fff";
  ctx.stroke();

  eraseBtn.disabled = false;
  hideBanner();
  showBanner("info", "Точка выбрана. Нажмите «Стереть», чтобы удалить объект.");
});

eraseBtn.addEventListener("click", async () => {
  if (!originalFile || !clickPoint) return;

  setProcessing(true, "Ищу объект...");
  hideBanner();

  try {
    const segmentForm = new FormData();
    segmentForm.append("file", originalFile);
    segmentForm.append("x", clickPoint.x);
    segmentForm.append("y", clickPoint.y);

    const segmentResponse = await fetch("/segment", { method: "POST", body: segmentForm });
    if (!segmentResponse.ok) {
      throw new Error(await describeErrorResponse(segmentResponse));
    }
    const maskBlob = await segmentResponse.blob();

    setProcessing(true, "Закрашиваю...");
    const inpaintForm = new FormData();
    inpaintForm.append("file", originalFile);
    inpaintForm.append("mask", maskBlob, "mask.png");

    const inpaintResponse = await fetch("/inpaint", { method: "POST", body: inpaintForm });
    if (!inpaintResponse.ok) {
      throw new Error(await describeErrorResponse(inpaintResponse));
    }
    const resultBlob = await inpaintResponse.blob();
    const resultUrl = URL.createObjectURL(resultBlob);

    // Скелетон остаётся на экране, пока браузер декодирует и отрисовывает
    // картинку — переключаемся на неё только по onload, чтобы шаг "Результат"
    // не открывался пустым на долю секунды.
    resultImg.hidden = true;
    downloadLink.hidden = true;
    startOverBtn.hidden = true;
    resultSkeleton.classList.add("visible");
    goToStep(3);

    resultImg.onload = () => {
      resultSkeleton.classList.remove("visible");
      resultImg.hidden = false;
      downloadLink.href = resultUrl;
      downloadLink.hidden = false;
      startOverBtn.hidden = false;
      panels.result.classList.remove("just-arrived");
      void panels.result.offsetWidth;
      panels.result.classList.add("just-arrived");
    };
    resultImg.src = resultUrl;

    const modelTrained = inpaintResponse.headers.get("X-Model-Trained") === "true";
    if (modelTrained) {
      showBanner("success", "Готово! Результат ниже.");
    } else {
      showBanner("warning", "Готово! Модель ещё не обучена — результат случайный, не осмысленный.");
    }
  } catch (error) {
    goToStep(2);
    showBanner("error", error.message);
  } finally {
    setProcessing(false, "Стереть");
  }
});
