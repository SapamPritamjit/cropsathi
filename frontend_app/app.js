/* AI4Seva frontend — vanilla JS, centralized API access */
"use strict";

/* ------------------------------------------------------------------ *
 * Configuration
 * ------------------------------------------------------------------ */

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const IMAGE_MAX_BYTES = 8 * 1024 * 1024; // mirrors backend IMAGE_MAX_BYTES default
const STT_MAX_BYTES = 25 * 1024 * 1024; // mirrors backend STT_MAX_UPLOAD_BYTES default
const IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"];
const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"];
const AUDIO_EXTENSIONS = [".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"];
const FALLBACK_CROPS = [
  { crop: "maize", display_name: "Maize" },
  { crop: "rice", display_name: "Rice" },
  { crop: "potato", display_name: "Potato" },
  { crop: "tomato", display_name: "Tomato" },
];
// Below this top-class confidence, show the ranked candidate classes instead of
// a single primary prediction. Adjust in one place. 0.75 === 75%.
const LOW_CONFIDENCE_THRESHOLD = 0.75;
const CANDIDATE_CLASS_COUNT = 3;

function resolveApiBaseUrl() {
  const configured =
    typeof window !== "undefined" && window.CROPSATHI_API_BASE_URL
      ? String(window.CROPSATHI_API_BASE_URL).trim()
      : "";
  const base = configured || DEFAULT_API_BASE_URL;
  return { base: base.replace(/\/+$/, ""), source: configured ? "window.CROPSATHI_API_BASE_URL" : "default" };
}

const API = resolveApiBaseUrl();

/* ------------------------------------------------------------------ *
 * API helper (all requests go through here)
 * ------------------------------------------------------------------ */

class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message || "Request failed");
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details === undefined ? null : details;
  }
}

async function apiRequest(path, options) {
  const opts = options || {};
  const method = opts.method || "GET";
  const url = new URL(API.base + path);

  if (opts.query) {
    Object.keys(opts.query).forEach(function (key) {
      const value = opts.query[key];
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });
  }

  const init = { method: method };
  if (opts.signal) init.signal = opts.signal;

  if (opts.json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  } else if (opts.formData !== undefined) {
    init.body = opts.formData; // browser sets multipart boundary
  }

  let response;
  try {
    response = await fetch(url.toString(), init);
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError(0, "network_error", "Could not reach the AI4Seva backend. Check that it is running and that CORS allows this page's origin.", null);
  }

  if (!response.ok) {
    let code = String(response.status);
    let message = "The request failed.";
    let details = null;
    const contentType = (response.headers.get("content-type") || "").toLowerCase();
    if (contentType.indexOf("application/json") !== -1) {
      try {
        const body = await response.json();
        if (body && body.error) {
          code = body.error.code || code;
          message = body.error.message || message;
          details = body.error.details !== undefined ? body.error.details : null;
        }
      } catch (parseErr) {
        /* keep defaults */
      }
    }
    throw new ApiError(response.status, code, message, details);
  }

  const expect = opts.expect || "json";
  if (expect === "blob") return response.blob();
  if (expect === "text") return response.text();

  const finalType = (response.headers.get("content-type") || "").toLowerCase();
  if (finalType.indexOf("application/json") === -1) return response.text();
  return response.json();
}

/* ------------------------------------------------------------------ *
 * DOM helpers (safe — textContent by default)
 * ------------------------------------------------------------------ */

function $(id) { return document.getElementById(id); }
function qs(selector, root) { return (root || document).querySelector(selector); }
function qsa(selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); }

function el(tag, props, children) {
  const node = document.createElement(tag);
  if (props) {
    Object.keys(props).forEach(function (key) {
      const value = props[key];
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key in node) node[key] = value;
      else node.setAttribute(key, value);
    });
  }
  if (children) {
    children.forEach(function (child) {
      if (child === null || child === undefined) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
  }
  return node;
}

/* ------------------------------------------------------------------ *
 * Application state
 * ------------------------------------------------------------------ */

const state = {
  voiceEnabled: true,
  crops: [],
  sessionHistory: [], // {crop, displayName, label, confidence, at}
  sessionCrops: [],   // {name, at}
  lastResult: null,
  lastSpeechText: null,
  selectedLeafFile: null,
  diagnosisRequestId: 0,
  insightSession: { crop: "", cropDisplay: "", turns: [], nextTurn: 1 },
  activeAudio: null,
  activeAudioUrl: null,
  modelsPayload: null,
  transcriptRecorder: null,
  transcriptChunks: [],
  transcriptStream: null,
  schemeController: null,
  insightController: null,
};

const VIEW_META = {
  dashboard: { title: "Dashboard", subtitle: "Overview and quick actions" },
  diagnosis: { title: "Crop Diagnosis", subtitle: "Screen a leaf photo for likely disease" },
  schemes: { title: "Scheme Finder", subtitle: "Search official agriculture schemes" },
  models: { title: "Models", subtitle: "Model capabilities and reported metrics" },
  crops: { title: "My Crops", subtitle: "Local demo list" },
  history: { title: "History", subtitle: "Predictions from this session" },
  insights: { title: "AI Crop Insights", subtitle: "Crop questions, prediction interpretation and charts" },
  support: { title: "Support", subtitle: "Help and contacts" },
  settings: { title: "Settings", subtitle: "Voice and connection" },
};

/* ------------------------------------------------------------------ *
 * Status messages
 * ------------------------------------------------------------------ */

function setStatus(id, kind, message) {
  const node = $(id);
  if (!node) return;
  if (!message) { node.hidden = true; node.textContent = ""; node.className = "form-status"; return; }
  node.hidden = false;
  node.className = "form-status is-" + kind;
  node.textContent = message;
}

const ERROR_MESSAGES = {
  network_error: "Could not reach the AI4Seva backend. Check that it is running and that CORS allows this page's origin.",
  model_unavailable: "This crop's model is not available on the backend.",
  model_load_failed: "This crop's model could not be loaded on the backend.",
  unsupported_crop: "That crop is not supported by the backend.",
  unsupported_family: "That model family is not supported by the backend.",
  fine_tuned_preprocessing_unverified: "Fine-tuned predictions are disabled on the backend (training preprocessing unverified).",
  unsupported_image_format: "That image format is not supported. Use JPEG, PNG, BMP, WEBP or TIFF.",
  corrupt_image: "The image could not be read. Try another photo.",
  empty_file: "The uploaded file is empty.",
  image_too_small: "The image is too small for the model.",
  image_too_large: "The image is too large for the model.",
  file_too_large: "The file is larger than the backend's upload limit.",
  inference_failed: "The backend model failed to produce a prediction.",
  tts_not_configured: "Speech synthesis is not configured on the backend (no API key).",
  tts_text_too_long: "The text is too long for the backend's speech model (200-character limit).",
  tts_rate_limited: "The speech service is rate limited. Please try again shortly.",
  tts_timeout: "The speech service timed out.",
  tts_upstream_error: "The speech service returned an error.",
  tts_empty_audio: "The speech service returned no audio.",
  stt_not_configured: "Speech-to-text is not configured on the backend (no API key).",
  stt_unsupported_media_type: "Unsupported audio type. Use wav, mp3, m4a, ogg, webm or flac.",
  stt_audio_too_large: "The audio clip is larger than the backend's limit.",
  stt_empty_audio: "The uploaded audio was empty.",
  stt_empty_transcription: "No speech could be transcribed from the audio.",
  stt_invalid_language: "The language hint must be a two-letter ISO-639-1 code.",
  stt_rate_limited: "The transcription service is rate limited. Please try again shortly.",
  stt_timeout: "The transcription service timed out.",
  stt_upstream_error: "The transcription service returned an error.",
  scheme_search_disabled: "Scheme search is disabled on the backend. Ask the operator to set SCHEME_SEARCH_ENABLED=true.",
  scheme_search_not_configured: "Scheme search is not configured on the backend (no search provider).",
  scheme_search_empty_query: "Provide a search term, or choose a crop or support need.",
  scheme_search_invalid_limit: "The result limit must be between 1 and 20.",
  scheme_search_invalid_crop: "That crop is not supported for scheme search.",
  scheme_search_invalid_need: "That support need is not recognised.",
  scheme_search_invalid_district: "The district value is too long.",
  scheme_search_invalid_language: "The language must be a two-letter code such as 'en'.",
  scheme_search_query_too_long: "The search text is too long (200-character limit).",
  scheme_search_timeout: "The scheme search timed out. Please try again.",
  scheme_search_upstream_error: "The upstream search provider returned an error. Please retry shortly.",
  insights_disabled: "AI crop Q&A is disabled on the backend. Ask the operator to set GROQ_INSIGHTS_ENABLED=true.",
  insights_not_configured: "AI crop Q&A is not configured on the backend (no API key). Ask the operator to set GROQ_API_KEY.",
  insight_empty_question: "Type a non-empty question.",
  insight_question_too_long: "The question is longer than the backend's limit. Please shorten it.",
  insight_invalid_topic: "The backend did not recognise that topic.",
  insight_invalid_context: "The backend rejected the prediction context.",
  insights_configuration_error: "AI crop Q&A is misconfigured on the backend.",
  insights_empty_answer: "The backend AI service returned an empty answer.",
  insights_timeout: "The backend AI service timed out. Please try again.",
  insights_rate_limited: "The backend AI service is rate limited. Please try again shortly.",
  insights_upstream_auth_error: "The backend AI service rejected the server credentials.",
  insights_upstream_unavailable: "The backend AI service could not be reached.",
  insights_upstream_error: "The backend AI service returned an error.",
};

function describeError(err) {
  if (!err) return "Something went wrong.";
  if (err.name === "AbortError") return null;
  if (err instanceof ApiError) {
    if (ERROR_MESSAGES[err.code]) return ERROR_MESSAGES[err.code];
    if (err.code === "validation_error") return "The request was rejected as invalid.";
    return err.message || "The request failed.";
  }
  return err.message || "Something went wrong.";
}

/* ------------------------------------------------------------------ *
 * Navigation
 * ------------------------------------------------------------------ */

function showView(name) {
  if (!VIEW_META[name]) name = "dashboard";
  qsa(".view").forEach(function (view) { view.classList.remove("is-active"); });
  const target = $("view-" + name);
  if (target) {
    target.classList.add("is-active");
    target.hidden = false;
  }
  qsa(".view").forEach(function (view) {
    if (view.id !== "view-" + name) view.hidden = true;
  });
  qsa(".nav-item").forEach(function (btn) {
    const active = btn.dataset.view === name;
    btn.classList.toggle("is-active", active);
    if (active) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
  const meta = VIEW_META[name];
  $("viewTitle").textContent = meta.title;
  $("viewSubtitle").textContent = meta.subtitle;
  stopSpeech();
  stopBrowserSpeech();
  closeSidebar();
  window.scrollTo({ top: 0 });
}

function openSidebar() {
  $("sidebar").classList.add("is-open");
  $("overlay").hidden = false;
  $("menuToggle").setAttribute("aria-expanded", "true");
}
function closeSidebar() {
  $("sidebar").classList.remove("is-open");
  $("overlay").hidden = true;
  $("menuToggle").setAttribute("aria-expanded", "false");
}

/* ------------------------------------------------------------------ *
 * Backend health + crops
 * ------------------------------------------------------------------ */

async function loadHealth() {
  const pill = $("healthPill");
  const text = $("healthText");
  pill.className = "status-pill is-wait";
  text.textContent = "Checking backend…";
  try {
    await apiRequest("/health");
    pill.className = "status-pill is-ok";
    text.textContent = "Backend online";
    return true;
  } catch (err) {
    pill.className = "status-pill is-err";
    text.textContent = "Backend offline";
    return false;
  }
}

async function loadCrops() {
  const diagnosisSelect = $("diagnosisCrop");
  const schemeSelect = $("schemeCrop");
  let crops = [];

  try {
    const data = await apiRequest("/crops");
    const list = (data && data.crops) || [];
    crops = list.map(function (entry) {
      return { crop: entry.crop, display_name: entry.display_name || entry.crop };
    });
  } catch (err) {
    crops = [];
  }

  const usedFallback = crops.length === 0;
  if (usedFallback) crops = FALLBACK_CROPS.slice();
  state.crops = crops;

  function fillSelect(select, placeholder) {
    if (!select) return;
    const previous = select.value;
    select.textContent = "";
    select.appendChild(el("option", { value: "", text: placeholder }));
    crops.forEach(function (entry) {
      select.appendChild(el("option", { value: entry.crop, text: entry.display_name }));
    });
    if (previous) select.value = previous;
  }

  fillSelect(diagnosisSelect, "Select a crop…");
  fillSelect(schemeSelect, "Any crop");
  fillSelect($("comparisonCrop"), "Select a crop…");
  fillSelect($("insightCrop"), "Select a crop…");
  fillSelect($("chartCrop"), "Select a crop…");
}

/* ------------------------------------------------------------------ *
 * Diagnosis
 * ------------------------------------------------------------------ */

let currentPreviewUrl = null;

function resetPreview() {
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
  }
  const img = $("previewImg");
  if (img) { img.onerror = null; img.onload = null; img.removeAttribute("src"); }
  $("previewWrap").hidden = true;
  $("previewMeta").textContent = "";
  state.selectedLeafFile = null;
}

function clearLeafInputs() {
  ["leafCamera", "leafUpload"].forEach(function (id) {
    const input = $(id);
    if (input) input.value = "";
  });
}

function validateFile(file) {
  if (!file || !file.size) return "Please choose an image file.";
  const name = (file.name || "").toLowerCase();
  const type = file.type || "";
  const knownType = IMAGE_MIME_TYPES.indexOf(type) !== -1;
  const knownExt = IMAGE_EXTENSIONS.some(function (ext) { return name.slice(-ext.length) === ext; });
  if (!knownType && !knownExt) {
    return "Unsupported file type. Use JPEG, PNG, BMP, WEBP or TIFF.";
  }
  if (file.size > IMAGE_MAX_BYTES) {
    return "This file is larger than 8 MB. Please choose a smaller image.";
  }
  return null;
}

/* Clean, display-only file labels (never a path or an object URL). */
function baseFileName(name) {
  return String(name || "image").split(/[\\/]/).pop();
}

function shortenFileName(name, max) {
  const s = baseFileName(name);
  const limit = max || 42;
  if (s.length <= limit) return s;
  const head = Math.ceil((limit - 1) / 2);
  const tail = limit - 1 - head;
  return s.slice(0, head) + "…" + s.slice(s.length - tail);
}

function formatFileSize(bytes) {
  if (typeof bytes !== "number" || bytes < 0 || !isFinite(bytes)) return "—";
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1024)) + " KB";
}

function renderPreviewMeta(file) {
  const meta = $("previewMeta");
  if (!meta) return;
  meta.textContent = "";
  if (!file) return;
  const crop = $("diagnosisCrop") ? $("diagnosisCrop").value : "";
  const cropName = crop ? cropDisplayName(crop) : "";
  if (cropName) {
    meta.appendChild(el("div", { class: "preview__row" }, [
      el("span", { class: "preview__key", text: "Crop" }),
      el("span", { class: "preview__value", text: cropName }),
    ]));
  }
  const fullName = baseFileName(file.name);
  const nameEl = el("div", { class: "preview__filename", text: shortenFileName(fullName) });
  nameEl.setAttribute("title", fullName);
  meta.appendChild(el("div", { class: "preview__row" }, [
    el("span", { class: "preview__key", text: "File" }),
    nameEl,
  ]));
  meta.appendChild(el("div", { class: "preview__row" }, [
    el("span", { class: "preview__key", text: "Size" }),
    el("span", { class: "preview__value", text: formatFileSize(file.size) }),
  ]));
}

function onFileSelected(event) {
  const input = event.target;
  const file = input.files && input.files[0];
  if (!file) return; // user cancelled or selected nothing

  const problem = validateFile(file);
  if (problem) {
    setStatus("diagnosisStatus", "error", problem);
    clearLeafInputs();
    resetPreview();
    return;
  }

  // Replace any previous preview and keep the two input paths from disagreeing.
  resetPreview();
  ["leafCamera", "leafUpload"].forEach(function (id) {
    if (id !== input.id) { const other = $(id); if (other) other.value = ""; }
  });
  setStatus("diagnosisStatus", null);

  state.selectedLeafFile = file;
  currentPreviewUrl = URL.createObjectURL(file);
  const img = $("previewImg");
  img.onerror = function () {
    clearLeafInputs();
    resetPreview();
    setStatus("diagnosisStatus", "error", "The image preview could not be displayed. Try another photo or file.");
  };
  img.onload = function () { img.onerror = null; img.onload = null; };
  img.src = currentPreviewUrl;
  renderPreviewMeta(file);
  $("previewWrap").hidden = false;
}

function showResultEmpty(title, hint) {
  const empty = $("resultEmpty");
  empty.hidden = false;
  empty.textContent = "";
  empty.appendChild(el("p", { text: title }));
  if (hint) empty.appendChild(el("p", { class: "muted-sm", text: hint }));
}

function clearResultForNewAnalysis() {
  // Drop the previous prediction so it can never be mistaken for the new one.
  state.lastResult = null;
  state.lastSpeechText = null;
  const body = $("resultBody");
  body.textContent = "";
  body.hidden = true;
  $("resetDiagnosisBtn").hidden = true;
  showResultEmpty("Analyzing new image…", "The previous result was cleared so it cannot be confused with this analysis.");
}

function resetDiagnosis() {
  state.diagnosisRequestId += 1; // invalidate any in-flight request
  $("diagnosisForm").reset();
  clearLeafInputs();
  resetPreview();
  setStatus("diagnosisStatus", null);
  $("resultBody").hidden = true;
  $("resultBody").textContent = "";
  showResultEmpty("No prediction yet.", "Select a crop and an image, then choose “Analyze leaf”.");
  $("resetDiagnosisBtn").hidden = true;
  $("analyzeBtn").disabled = false;
  $("diagnosisCrop").disabled = false;
  $("takePhotoBtn").disabled = false;
  $("uploadPhotoBtn").disabled = false;
  state.lastResult = null;
  state.lastSpeechText = null;
}

function validatePredictionPayload(result) {
  if (!result || typeof result !== "object") {
    return "The backend returned an empty response. Please try again.";
  }
  if (!result.predicted_label || !Array.isArray(result.probabilities) || !result.probabilities.length) {
    return "The backend returned an incomplete prediction. Please try again.";
  }
  return null;
}

function formatProbability(value) {
  return (value * 100).toFixed(1) + "%";
}

// Validate + rank probability entries without mutating the response array.
// Returns a NEW array of {index, label, probability} sorted by descending
// probability; entries with a missing/non-finite/negative probability are
// dropped so invalid values can never be sorted or rendered as if real.
function validProbabilityEntries(probabilities) {
  if (!Array.isArray(probabilities)) return [];
  return probabilities
    .map(function (item, i) {
      const source = item && typeof item === "object" ? item : {};
      const prob = typeof source.probability === "number" ? source.probability : NaN;
      const valid = isFinite(prob) && prob >= 0;
      return {
        index: source.index != null ? source.index : i,
        label: source.label ? String(source.label) : "Class " + i,
        probability: valid ? prob : null,
        valid: valid,
      };
    })
    .filter(function (entry) { return entry.valid; })
    .sort(function (a, b) { return b.probability - a.probability; });
}

function renderProbabilityList(probabilities) {
  const list = el("ul", { class: "prob-list" });
  const entries = Array.isArray(probabilities) ? probabilities : [];
  entries.forEach(function (item) {
    const source = item && typeof item === "object" ? item : {};
    const prob = typeof source.probability === "number" && isFinite(source.probability) && source.probability >= 0
      ? source.probability
      : null;
    const pctText = prob !== null ? formatProbability(prob) : "—";
    const bar = el("span", { class: "prob-bar" }, [el("span", {})]);
    if (prob !== null) bar.firstChild.style.width = pctText;
    list.appendChild(el("li", { class: "prob-row" }, [
      el("span", { class: "prob-row__label", text: source.label ? String(source.label) : "Unknown" }),
      el("span", { class: "prob-row__value", text: pctText }),
      bar,
    ]));
  });
  return list;
}

// Low-confidence presentation: leading candidate + ranked alternatives + a
// plain-language uncertainty note. All values come from the backend's
// probability distribution; nothing is invented or relabelled.
function renderCandidateResult(body, result, candidates) {
  const lead = candidates[0];
  body.appendChild(el("p", { class: "result-crop-label", text: "Most likely" }));
  body.appendChild(el("h4", { class: "result-label", text: lead.label || result.predicted_label || "Unknown" }));
  body.appendChild(el("div", { class: "result-confidence" }, [
    el("strong", { text: formatProbability(lead.probability) }),
    el("span", { class: "muted-sm", text: "top model score" }),
  ]));

  body.appendChild(el("p", { class: "muted-sm", text: "Top " + candidates.length + " possible classes, ranked by model score:" }));
  const list = el("ul", { class: "candidate-list" });
  candidates.forEach(function (item, i) {
    const pctText = formatProbability(item.probability);
    const bar = el("span", { class: "prob-bar" }, [el("span", { style: "width:" + pctText + ";" })]);
    list.appendChild(el("li", { class: "candidate" + (i === 0 ? " candidate--lead" : "") }, [
      el("span", { class: "candidate__rank", text: "#" + (i + 1) }),
      el("span", { class: "candidate__label", text: item.label }),
      el("span", { class: "candidate__value", text: pctText }),
      bar,
    ]));
  });
  body.appendChild(list);

  const word = candidates.length === 3 ? "three" : String(candidates.length);
  body.appendChild(el("p", {
    class: "result-uncertainty",
    text: "Lower model confidence: these are the " + word + " highest-scoring classes, not confirmed diagnoses. " +
      "Compare the symptoms and consider a clearer photo or advice from a local agricultural expert.",
  }));
}

function renderPrediction(result) {
  const body = $("resultBody");
  body.textContent = "";
  $("resultEmpty").hidden = true;
  body.hidden = false;

  const hasConfidence = typeof result.confidence === "number" && isFinite(result.confidence);
  const confidencePct = hasConfidence ? formatProbability(result.confidence) : "—";
  const candidates = validProbabilityEntries(result.probabilities);
  const topCandidates = candidates.slice(0, CANDIDATE_CLASS_COUNT);
  const isLowConfidence = hasConfidence
    && result.confidence < LOW_CONFIDENCE_THRESHOLD
    && topCandidates.length > 0;

  body.appendChild(el("p", { class: "result-crop", text: result.display_name || result.crop || "" }));

  if (isLowConfidence) {
    renderCandidateResult(body, result, topCandidates);
  } else {
    body.appendChild(el("h4", { class: "result-label", text: result.predicted_label || "Unknown" }));
    body.appendChild(el("div", { class: "result-confidence" }, [
      el("strong", { text: confidencePct }),
      el("span", { class: "muted-sm", text: "model confidence" }),
    ]));
  }

  // Technical model metadata (family/implementation/classes/input size/weights/
  // label source/predicted index) is deliberately not shown in the normal
  // result. The backend response stays intact; nothing is removed or altered.

  const probs = Array.isArray(result.probabilities) ? result.probabilities : [];
  if (probs.length) {
    body.appendChild(el("p", { class: "muted-sm", text: "Class probabilities (all " + probs.length + "):" }));
    body.appendChild(renderProbabilityList(probs));
  }

  // Farmer-friendly summary text (from POST /speech/response) + playback controls.
  const speechBlock = el("div", { class: "speech-block" }, [
    el("div", { class: "speech-block__label", text: "Spoken summary" }),
    el("p", { class: "speech-block__text", id: "speechTextEl", text: "Preparing summary…" }),
  ]);
  body.appendChild(speechBlock);

  const actions = el("div", { class: "form-actions" }, [
    el("button", { class: "btn btn-secondary", type: "button", id: "speakResultBtn", text: "Listen to result" }),
    el("button", { class: "btn btn-ghost", type: "button", id: "stopSpeakBtn", text: "Stop" }),
    el("button", { class: "btn btn-primary", type: "button", id: "exploreResultBtn", text: "Continue exploring this crop" }),
  ]);
  body.appendChild(actions);
  $("speakResultBtn").addEventListener("click", function () { speakResult(); });
  $("stopSpeakBtn").addEventListener("click", stopSpeech);
  $("exploreResultBtn").addEventListener("click", exploreResultFromDiagnosis);

  body.appendChild(el("div", {
    class: "disclaimer",
    text: result.disclaimer || "This is a model prediction, not a confirmed diagnosis. Consult your local agriculture extension officer before acting.",
  }));
}

async function submitDiagnosis(event) {
  event.preventDefault();
  const cropSelect = $("diagnosisCrop");
  const crop = cropSelect.value;
  const file = state.selectedLeafFile;

  if (!crop) { setStatus("diagnosisStatus", "error", "Please select a crop."); return; }
  const problem = validateFile(file);
  if (problem) { setStatus("diagnosisStatus", "error", problem); return; }

  // Tag this request so a stale response cannot overwrite a newer prediction.
  const requestId = ++state.diagnosisRequestId;

  setStatus("diagnosisStatus", "info", "Analyzing image…");
  const analyzeBtn = $("analyzeBtn");
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  cropSelect.disabled = true;
  $("takePhotoBtn").disabled = true;
  $("uploadPhotoBtn").disabled = true;

  clearResultForNewAnalysis();

  const formData = new FormData();
  formData.append("crop", crop);
  formData.append("file", file, file.name);
  formData.append("family", "baseline");

  try {
    const result = await apiRequest("/predict", { method: "POST", formData: formData });
    if (requestId !== state.diagnosisRequestId) return; // superseded
    const invalid = validatePredictionPayload(result);
    if (invalid) { setStatus("diagnosisStatus", "error", invalid); return; }
    setStatus("diagnosisStatus", "ok", "Prediction ready.");
    renderPrediction(result);
    $("resetDiagnosisBtn").hidden = false;

    const cropName = result.display_name || crop;
    state.lastResult = result;
    state.sessionHistory.unshift({
      crop: result.crop || crop,
      displayName: cropName,
      label: result.predicted_label || "Unknown",
      confidence: result.confidence,
      at: new Date(),
    });
    renderHistory();
    renderDashboardSession();

    await prepareSpeech(result, requestId);
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (requestId !== state.diagnosisRequestId) return; // superseded
    setStatus("diagnosisStatus", "error", describeError(err));
  } finally {
    if (requestId === state.diagnosisRequestId) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze leaf";
      cropSelect.disabled = false;
      $("takePhotoBtn").disabled = false;
      $("uploadPhotoBtn").disabled = false;
    }
  }
}

/* ------------------------------------------------------------------ *
 * Voice / speech
 * ------------------------------------------------------------------ */

function setVoice(enabled) {
  state.voiceEnabled = !!enabled;
  $("voiceToggle").checked = state.voiceEnabled;
  try { localStorage.setItem("ai4seva.voice", state.voiceEnabled ? "1" : "0"); } catch (e) { /* ignore */ }
  if (!state.voiceEnabled) { stopSpeech(); stopBrowserSpeech(); }
}

function stopSpeech() {
  if (state.activeAudio) {
    try { state.activeAudio.pause(); } catch (e) { /* ignore */ }
    state.activeAudio = null;
  }
  if (state.activeAudioUrl) {
    URL.revokeObjectURL(state.activeAudioUrl);
    state.activeAudioUrl = null;
  }
  cancelSpeech();
}

function browserSpeak(text) {
  if (!ttsSupported()) return false;
  if (!tts.voicesReady) loadVoices();
  try {
    const utterance = new SpeechSynthesisUtterance(text);
    if (tts.voice) {
      utterance.voice = tts.voice;
      utterance.lang = tts.voice.lang || "en";
    } else {
      utterance.lang = (navigator.language || "en").replace("_", "-");
    }
    window.speechSynthesis.speak(utterance);
    return true;
  } catch (e) {
    return false;
  }
}

async function playBackendSpeech(text) {
  const blob = await apiRequest("/speech/speak", {
    method: "POST",
    json: { text: text },
    expect: "blob",
  });
  stopSpeech();
  state.activeAudioUrl = URL.createObjectURL(blob);
  const audio = new Audio(state.activeAudioUrl);
  state.activeAudio = audio;
  audio.addEventListener("ended", function () {
    if (state.activeAudioUrl) { URL.revokeObjectURL(state.activeAudioUrl); state.activeAudioUrl = null; }
    state.activeAudio = null;
  });
  await audio.play();
}

async function speakText(text) {
  if (!text) return;
  if (!state.voiceEnabled) {
    setStatus("diagnosisStatus", "warn", "Voice output is off — turn it on in Settings.");
    return;
  }
  setStatus("diagnosisStatus", "info", "Preparing audio…");
  try {
    await playBackendSpeech(text);
    setStatus("diagnosisStatus", "ok", "Playing audio.");
  } catch (err) {
    if (err && err.name === "AbortError") return;
    // Graceful fallback to browser speech synthesis.
    if (browserSpeak(text)) {
      setStatus("diagnosisStatus", "warn", "Backend audio unavailable — using browser voice.");
    } else {
      setStatus("diagnosisStatus", "error", describeError(err) || "Audio playback failed.");
    }
  }
}

async function prepareSpeech(result, requestId) {
  const textEl = $("speechTextEl");
  try {
    const data = await apiRequest("/speech/response", {
      method: "POST",
      json: {
        crop: result.crop || null,
        display_name: result.display_name || null,
        predicted_label: result.predicted_label || null,
        confidence: typeof result.confidence === "number" ? result.confidence : null,
      },
    });
    if (requestId !== undefined && requestId !== state.diagnosisRequestId) return; // superseded
    state.lastSpeechText = data && data.text ? data.text : null;
  } catch (err) {
    if (requestId !== undefined && requestId !== state.diagnosisRequestId) return; // superseded
    state.lastSpeechText = null;
    if (textEl) textEl.textContent = describeError(err) || "Summary unavailable.";
    return;
  }
  if (textEl) {
    textEl.textContent = state.lastSpeechText || "Summary unavailable.";
  }
}

function speakResult() {
  if (state.lastSpeechText) { speakText(state.lastSpeechText); return; }
  if (state.lastResult) {
    const r = state.lastResult;
    const pct = typeof r.confidence === "number" ? (r.confidence * 100).toFixed(0) + "%" : "";
    speakText((r.display_name || r.crop) + " prediction: " + r.predicted_label + " " + pct + ". This is not a confirmed diagnosis.");
  }
}

/* Page-wide browser text-to-speech (distinct from backend /speech/speak WAV). */

const tts = {
  state: "stopped", // stopped | playing | paused
  mode: "",         // "" | "page" | "section" (what the current read covers)
  chunks: [],
  index: 0,
  generation: 0,
  voice: null,
  voicesReady: false,
};

function ttsSupported() {
  return typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    typeof window.SpeechSynthesisUtterance !== "undefined";
}

function pickVoice(voices) {
  if (!voices || !voices.length) return null;
  const preferred = ["en-in", "en-gb", "en-us", "en"];
  for (let p = 0; p < preferred.length; p++) {
    for (let i = 0; i < voices.length; i++) {
      const lang = String(voices[i].lang || "").replace("_", "-").toLowerCase();
      if (lang === preferred[p] || lang.indexOf(preferred[p] + "-") === 0) return voices[i];
    }
  }
  for (let i = 0; i < voices.length; i++) {
    if (/^en/.test(String(voices[i].lang || "").toLowerCase())) return voices[i];
  }
  return voices[0] || null;
}

function loadVoices() {
  if (!ttsSupported()) return;
  let voices = [];
  try { voices = window.speechSynthesis.getVoices() || []; } catch (e) { voices = []; }
  if (!voices.length) return;
  tts.voicesReady = true;
  tts.voice = pickVoice(voices);
}

function initTTSVoices() {
  if (!ttsSupported()) return;
  loadVoices();
  try {
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
  } catch (e) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }
}

function updateTTSControls() {
  const pauseBtn = $("ttsPauseBtn");
  const stopBtn = $("ttsStopBtn");
  const statusEl = $("ttsStatus");
  if (!pauseBtn || !stopBtn || !statusEl) return;
  const active = tts.state !== "stopped";
  pauseBtn.hidden = !active;
  stopBtn.hidden = !active;
  pauseBtn.textContent = tts.state === "paused" ? "Resume" : "Pause";
  if (active) {
    statusEl.hidden = false;
    if (tts.state === "paused") {
      statusEl.textContent = "Paused (browser voice)";
    } else {
      statusEl.textContent = tts.mode === "section"
        ? "Reading section (browser voice)"
        : "Reading page (browser voice)";
    }
  } else {
    statusEl.hidden = true;
    statusEl.textContent = "";
  }
}

function setTTSNotice(message) {
  const statusEl = $("ttsStatus");
  if (!statusEl) return;
  statusEl.hidden = false;
  statusEl.textContent = message;
  const pauseBtn = $("ttsPauseBtn");
  const stopBtn = $("ttsStopBtn");
  if (pauseBtn) pauseBtn.hidden = true;
  if (stopBtn) stopBtn.hidden = true;
}

function collectReadableText(root) {
  if (!root) return "";
  const clone = root.cloneNode(true);
  const strip = clone.querySelectorAll('[aria-hidden="true"], button, select, input, textarea, [hidden], script, style, .no-read');
  Array.prototype.forEach.call(strip, function (node) {
    if (node.parentNode) node.parentNode.removeChild(node);
  });
  return (clone.textContent || "").replace(/\s+/g, " ").trim();
}

function splitIntoChunks(text, maxLen) {
  const limit = maxLen || 200;
  const sentences = text.match(/[^.!?]+[.!?]*/g) || [text];
  const chunks = [];
  let current = "";
  sentences.forEach(function (sentence) {
    const s = sentence.trim();
    if (!s) return;
    if (current && (current.length + 1 + s.length) > limit) {
      chunks.push(current);
      current = s;
    } else {
      current = current ? current + " " + s : s;
    }
  });
  if (current) chunks.push(current);
  return chunks;
}

function cancelSpeech() {
  if (ttsSupported()) {
    try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    // cancel() can leave the engine in a paused state on some browsers; clear it
    // or every later speak() call is queued and never played.
    try { if (window.speechSynthesis.paused) window.speechSynthesis.resume(); } catch (e) { /* ignore */ }
  }
  tts.generation += 1;
}

function speakTTSChunk() {
  if (tts.state === "stopped") return;
  if (tts.index >= tts.chunks.length) { finishBrowserRead(); return; }

  const gen = tts.generation;
  const utterance = new SpeechSynthesisUtterance(tts.chunks[tts.index]);
  if (tts.voice) {
    utterance.voice = tts.voice;
    utterance.lang = tts.voice.lang || "en";
  } else {
    utterance.lang = (navigator.language || "en").replace("_", "-");
  }
  utterance.rate = 1;
  utterance.pitch = 1;

  utterance.onend = function () {
    if (gen !== tts.generation || tts.state === "stopped") return;
    tts.index += 1;
    speakTTSChunk();
  };
  utterance.onerror = function (event) {
    if (gen !== tts.generation) return;
    const code = event && event.error ? event.error : "";
    if (code === "interrupted" || code === "canceled") return;
    tts.state = "stopped";
    tts.chunks = [];
    tts.index = 0;
    updateTTSControls();
    setTTSNotice("Browser speech failed (" + (code || "unknown error") + ").");
  };

  try {
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    tts.state = "stopped";
    updateTTSControls();
    setTTSNotice("Browser speech could not start.");
  }
}

function finishBrowserRead() {
  tts.state = "stopped";
  tts.chunks = [];
  tts.index = 0;
  updateTTSControls();
  setTTSNotice("Finished reading.");
  const gen = tts.generation;
  window.setTimeout(function () {
    if (tts.generation === gen && tts.state === "stopped") {
      tts.mode = "";
      const statusEl = $("ttsStatus");
      if (statusEl) { statusEl.hidden = true; statusEl.textContent = ""; }
    }
  }, 2500);
}

function startBrowserRead() {
  beginRead(qs(".view.is-active"), "page");
}

/* Read a single section (card, chart, panel, turn) — same engine/queue. */
function startSectionRead(section) {
  beginRead(section, "section");
}

function beginRead(root, mode) {
  if (!ttsSupported()) {
    setTTSNotice("This browser does not support speech synthesis.");
    return;
  }
  if (!state.voiceEnabled) {
    setTTSNotice("Voice output is off — turn it on in Settings.");
    return;
  }
  // Starting either mode cancels whatever was playing.
  cancelSpeech();

  const text = collectReadableText(root);
  if (!text) {
    setTTSNotice("Nothing to read in this section.");
    return;
  }

  if (!tts.voicesReady) loadVoices();
  if (!tts.voicesReady) {
    setTTSNotice("Loading browser voices… press again in a moment.");
    return;
  }

  tts.mode = mode === "section" ? "section" : "page";
  tts.chunks = splitIntoChunks(text);
  tts.index = 0;
  tts.state = "playing";
  updateTTSControls();
  // Start on a fresh task: some browsers drop speak() issued right after cancel().
  window.setTimeout(function () {
    if (tts.state === "playing") speakTTSChunk();
  }, 80);
}

function togglePauseRead() {
  if (!ttsSupported() || tts.state === "stopped") return;
  if (tts.state === "playing") {
    try { window.speechSynthesis.pause(); } catch (e) { /* ignore */ }
    tts.state = "paused";
  } else if (tts.state === "paused") {
    try { window.speechSynthesis.resume(); } catch (e) { /* ignore */ }
    tts.state = "playing";
  }
  updateTTSControls();
}

function stopBrowserSpeech() {
  cancelSpeech();
  tts.chunks = [];
  tts.index = 0;
  tts.state = "stopped";
  tts.mode = "";
  updateTTSControls();
}

/* ---- Section-level read controls ---------------------------------- *
 * Each readable section (card, chart, insight turn) gets a small
 * speaker button that reads only that section. The button is a real
 * <button> built with DOM APIs, so keyboard activation works and it is
 * excluded from collected text (the collector strips buttons).
 * ------------------------------------------------------------------ */

const READ_SECTION_SELECTORS = [".hero", ".card", ".turn", ".chart"];

function speakerIcon() {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  function path(d, opts) {
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", d);
    p.setAttribute("fill", (opts && opts.fill) || "none");
    if (opts && opts.stroke) {
      p.setAttribute("stroke", opts.stroke);
      p.setAttribute("stroke-width", opts.width || "1.7");
      p.setAttribute("stroke-linecap", "round");
    }
    return p;
  }
  svg.appendChild(path("M4 9h4l5-4v14l-5-4H4z", { fill: "currentColor" }));
  svg.appendChild(path("M15.5 8.5a5 5 0 0 1 0 7", { stroke: "currentColor" }));
  svg.appendChild(path("M18 6a8.5 8.5 0 0 1 0 12", { stroke: "currentColor" }));
  return svg;
}

function addSectionReader(section) {
  if (!section || section.dataset.readReady === "1") return;
  section.dataset.readReady = "1";

  const btn = el("button", { class: "section-read__btn", type: "button" });
  btn.setAttribute("aria-label", "Read this section aloud");
  btn.setAttribute("title", "Read this section aloud");
  btn.appendChild(speakerIcon());
  btn.addEventListener("click", function () { startSectionRead(section); });

  const hook = qs(".card__head, .turn__head, .model-card__top, .scheme-card__top", section);
  if (hook) {
    btn.classList.add("section-read__btn--inline");
    const last = hook.lastElementChild;
    // Keep any existing right-hand item (badge, Reload, meta) next to the
    // speaker control instead of letting space-between separate them.
    if (last && hook.children.length >= 2) {
      const group = el("div", { class: "section-read__group" });
      hook.replaceChild(group, last);
      group.appendChild(last);
      group.appendChild(btn);
    } else {
      hook.appendChild(btn);
    }
  } else {
    const wrap = el("div", { class: "section-read" });
    wrap.appendChild(btn);
    section.classList.add("has-section-read");
    section.insertBefore(wrap, section.firstChild);
  }
}

function attachSectionReaders() {
  const main = $("main-content");
  if (!main) return;
  READ_SECTION_SELECTORS.forEach(function (selector) {
    qsa(selector, main).forEach(function (section) { addSectionReader(section); });
  });
}

let readerObserver = null;
function initSectionReaders() {
  const main = $("main-content");
  if (!main) return;
  attachSectionReaders();
  if (typeof MutationObserver === "undefined") return;
  readerObserver = new MutationObserver(function () {
    // Re-scan after dynamic content changes; disconnect while injecting so
    // our own buttons do not re-trigger the observer in a loop.
    if (readerObserver) readerObserver.disconnect();
    attachSectionReaders();
    readerObserver.observe(main, { childList: true, subtree: true });
  });
  readerObserver.observe(main, { childList: true, subtree: true });
}

/* ------------------------------------------------------------------ *
 * Scheme finder
 * ------------------------------------------------------------------ */

function clearSchemeResults() {
  $("schemeResults").textContent = "";
  $("schemeRetrieved").textContent = "";
  const summary = $("schemeSummary");
  if (summary) { summary.textContent = ""; summary.hidden = true; }
}

function renderSchemeEmpty(message) {
  const wrap = $("schemeResults");
  wrap.textContent = "";
  wrap.appendChild(el("article", { class: "card" }, [
    el("h3", { class: "card__title", text: "No results" }),
    el("p", { class: "muted-sm", text: message }),
  ]));
}

function sourceBadgeClass(result) {
  if (result.verified && result.source_type === "official_manipur") return "badge-official";
  if (result.verified && result.source_type === "official_goi") return "badge-official";
  if (result.source_type === "other") return "badge-other";
  return "badge-unverified";
}

function sourceTypeLabel(result) {
  if (result.source_type === "official_manipur") return "Official · Manipur";
  if (result.source_type === "official_goi") return "Official · Govt of India";
  return "Other source";
}

function renderSchemeResults(payload) {
  const wrap = $("schemeResults");
  wrap.textContent = "";
  const results = (payload && payload.results) || [];

  if (payload && payload.retrieved_at) {
    $("schemeRetrieved").textContent = "Retrieved " + payload.retrieved_at;
  } else {
    $("schemeRetrieved").textContent = "";
  }

  const summary = $("schemeSummary");
  if (summary) {
    const total = payload && typeof payload.result_count === "number" ? payload.result_count : results.length;
    const official = payload && typeof payload.official_count === "number" ? payload.official_count : 0;
    summary.hidden = false;
    summary.textContent = total + (total === 1 ? " result" : " results") + " · " + official + " official · language " + ((payload && payload.language) || "en");
  }

  if (!results.length) {
    renderSchemeEmpty("No schemes matched your search. Try broader terms, a different crop, or remove filters.");
  }

  results.forEach(function (result) {
    const card = el("article", { class: "card scheme-card" });

    const title = el("span", { class: "scheme-card__title", text: result.title || "Untitled" });
    const verifiedBadge = el("span", {
      class: "badge " + (result.verified ? "badge-verified" : "badge-unverified"),
      text: result.verified ? "Verified domain" : "Unverified",
    });
    card.appendChild(el("div", { class: "scheme-card__top" }, [title, verifiedBadge]));

    if (result.source_domain) {
      card.appendChild(el("span", { class: "scheme-card__domain", text: result.source_domain }));
    }

    if (result.snippet) {
      card.appendChild(el("p", { class: "scheme-card__snippet", text: result.snippet }));
    }

    const typeBadge = el("span", { class: "badge " + sourceBadgeClass(result), text: sourceTypeLabel(result) });
    const link = el("a", {
      class: "scheme-card__link",
      href: result.url,
      target: "_blank",
      rel: "noopener noreferrer",
      text: "Open official page ↗",
    });
    card.appendChild(el("div", { class: "scheme-card__foot" }, [typeBadge, link]));

    wrap.appendChild(card);
  });

  if (payload && payload.disclaimer) {
    wrap.appendChild(el("div", { class: "disclaimer", text: payload.disclaimer }));
  }
}

function currentSchemeMode() {
  const checked = qs('input[name="schemeMode"]:checked');
  return checked && checked.value === "custom" ? "custom" : "topic";
}

function schemeQueryText() {
  if (currentSchemeMode() === "custom") return $("schemeQuery").value.trim();
  const topicSelect = $("schemeTopic");
  return topicSelect ? topicSelect.value.trim() : "";
}

function setSchemeMode(mode) {
  const isCustom = mode === "custom";
  const topicField = $("schemeTopicField");
  const queryField = $("schemeQueryField");
  if (topicField) topicField.hidden = isCustom;
  if (queryField) queryField.hidden = !isCustom;
}

function onSchemeModeChange(event) {
  setSchemeMode(event.target.value);
}

function fillQueryFromForm() {
  const query = {};
  const q = schemeQueryText();
  const crop = $("schemeCrop").value;
  const need = $("schemeNeed").value;
  const district = $("schemeDistrict").value.trim();
  const limit = $("schemeLimit").value;
  const language = $("schemeLanguage").value;
  if (q) query.q = q;
  if (crop) query.crop = crop;
  if (need) query.need = need;
  if (district) query.district = district;
  if (limit) query.limit = limit;
  if (language) query.language = language;
  return query;
}

async function submitSchemeSearch(event) {
  event.preventDefault();
  const query = fillQueryFromForm();

  if (!query.q && !query.crop && !query.need) {
    setStatus("schemeStatus", "warn", "Enter a search term, or choose a crop or support need.");
    return;
  }

  if (state.schemeController) state.schemeController.abort();
  state.schemeController = new AbortController();

  clearSchemeResults();
  setStatus("schemeStatus", "info", "Searching schemes…");
  const button = $("schemeSearchBtn");
  button.disabled = true;
  button.textContent = "Searching…";

  try {
    const payload = await apiRequest("/schemes/search", {
      query: query,
      signal: state.schemeController.signal,
    });
    setStatus("schemeStatus", null);
    renderSchemeResults(payload);
  } catch (err) {
    if (err && err.name === "AbortError") return;
    clearSchemeResults();
    const kind = (err instanceof ApiError && (err.code === "scheme_search_disabled" || err.code === "scheme_search_not_configured"))
      ? "warn"
      : "error";
    setStatus("schemeStatus", kind, describeError(err));
  } finally {
    state.schemeController = null;
    button.disabled = false;
    button.textContent = "Search schemes";
  }
}

function clearSchemeForm() {
  $("schemeForm").reset();
  setSchemeMode("topic");
  clearSchemeResults();
  setStatus("schemeStatus", null);
}

/* ------------------------------------------------------------------ *
 * Session crops + history + dashboard summary
 * ------------------------------------------------------------------ */

function renderMyCrops() {
  const list = $("cropList");
  list.textContent = "";
  if (!state.sessionCrops.length) {
    list.appendChild(el("p", { class: "muted-sm", text: "No crops added in this session." }));
    return;
  }
  state.sessionCrops.forEach(function (item, index) {
    const row = el("div", { class: "session-item" }, [
      el("div", null, [
        el("div", { text: item.name }),
        el("div", { class: "session-item__meta", text: "Added " + item.at.toLocaleTimeString() }),
      ]),
      el("button", { class: "btn btn-ghost", type: "button", text: "Remove" }),
    ]);
    row.lastChild.addEventListener("click", function () {
      state.sessionCrops.splice(index, 1);
      renderMyCrops();
    });
    list.appendChild(row);
  });
}

function renderHistory() {
  const list = $("historyList");
  list.textContent = "";
  if (!state.sessionHistory.length) {
    list.appendChild(el("p", { class: "muted-sm", text: "No predictions yet in this session." }));
    return;
  }
  state.sessionHistory.forEach(function (item) {
    const pct = typeof item.confidence === "number" ? (item.confidence * 100).toFixed(1) + "%" : "—";
    list.appendChild(el("div", { class: "session-item" }, [
      el("div", null, [
        el("div", { text: item.displayName + " — " + item.label }),
        el("div", { class: "session-item__meta", text: item.at.toLocaleTimeString() + " · confidence " + pct }),
      ]),
    ]));
  });
}

function renderDashboardSession() {
  const summary = $("dashRecentSummary");
  if (!summary) return;
  summary.textContent = "";
  if (!state.sessionHistory.length) {
    const startBtn = el("button", { class: "btn btn-secondary", type: "button", text: "Start your first diagnosis" });
    startBtn.addEventListener("click", function () { showView("diagnosis"); });
    summary.appendChild(el("div", { class: "empty-state" }, [
      el("p", { class: "muted-sm", text: "No predictions yet in this session." }),
      startBtn,
    ]));
    return;
  }
  state.sessionHistory.slice(0, 3).forEach(function (item) {
    summary.appendChild(el("div", { class: "session-item" }, [
      el("div", null, [
        el("div", { text: item.displayName + " — " + item.label }),
        el("div", { class: "session-item__meta", text: item.at.toLocaleTimeString() }),
      ]),
    ]));
  });
}

function clearSession() {
  state.sessionHistory = [];
  state.sessionCrops = [];
  renderHistory();
  renderMyCrops();
  renderDashboardSession();
}

/* ------------------------------------------------------------------ *
 * Voice note transcription (backend)
 * ------------------------------------------------------------------ */

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("transcribeStatus", "error", "Recording is not supported in this browser.");
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    setStatus("transcribeStatus", "error", "MediaRecorder is not supported in this browser.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.transcriptStream = stream;
    state.transcriptChunks = [];
    const recorder = new MediaRecorder(stream);
    state.transcriptRecorder = recorder;
    recorder.addEventListener("dataavailable", function (event) {
      if (event.data && event.data.size > 0) state.transcriptChunks.push(event.data);
    });
    recorder.addEventListener("stop", onRecordingStopped);
    recorder.start();
    $("recordBtn").hidden = true;
    $("stopRecordBtn").hidden = false;
    setStatus("transcribeStatus", "info", "Recording… press Stop to transcribe.");
  } catch (err) {
    setStatus("transcribeStatus", "error", "Microphone permission was denied or unavailable.");
  }
}

function stopRecording() {
  if (state.transcriptRecorder && state.transcriptRecorder.state !== "inactive") {
    state.transcriptRecorder.stop();
  }
}

async function onRecordingStopped() {
  $("recordBtn").hidden = false;
  $("stopRecordBtn").hidden = true;
  if (state.transcriptStream) {
    state.transcriptStream.getTracks().forEach(function (track) { track.stop(); });
    state.transcriptStream = null;
  }
  const blob = new Blob(state.transcriptChunks, { type: "audio/webm" });
  state.transcriptChunks = [];
  state.transcriptRecorder = null;
  if (!blob.size) { setStatus("transcribeStatus", "error", "No audio was captured."); return; }
  transcribeBlob(blob, "recording.webm");
}

function validateAudioFile(file) {
  if (!file) return "Please choose an audio file.";
  const name = file.name.toLowerCase();
  const okExt = AUDIO_EXTENSIONS.some(function (ext) { return name.slice(-ext.length) === ext; });
  const okType = (file.type || "").indexOf("audio/") === 0 || file.type === "video/mp4" || file.type === "video/webm";
  if (!okExt && !okType) return "Unsupported audio type. Use wav, mp3, m4a, ogg, webm or flac.";
  if (file.size > STT_MAX_BYTES) return "This audio file is larger than 25 MB.";
  return null;
}

function onTranscribeFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const problem = validateAudioFile(file);
  if (problem) {
    setStatus("transcribeStatus", "error", problem);
    event.target.value = "";
    return;
  }
  $("transcriptOut").hidden = true;
  transcribeBlob(file, file.name);
}

async function transcribeBlob(blob, filename) {
  setStatus("transcribeStatus", "info", "Transcribing…");
  const language = $("transcribeLanguage") ? $("transcribeLanguage").value : "";
  const formData = new FormData();
  formData.append("file", blob, filename || "audio.webm");
  if (language) formData.append("language", language);
  try {
    const data = await apiRequest("/speech/transcribe", { method: "POST", formData: formData });
    const out = $("transcriptOut");
    out.hidden = false;
    out.textContent = (data && data.text) ? data.text : "(empty transcription)";
    if (data && data.language) {
      setStatus("transcribeStatus", "ok", "Transcription complete (detected language: " + data.language + ").");
    } else {
      setStatus("transcribeStatus", "ok", "Transcription complete.");
    }
  } catch (err) {
    setStatus("transcribeStatus", "error", describeError(err));
  }
}

/* ------------------------------------------------------------------ *
 * Feedback (local only)
 * ------------------------------------------------------------------ */

async function copyFeedback() {
  const text = $("feedbackText").value.trim();
  if (!text) { setStatus("feedbackStatus", "warn", "Nothing to copy yet."); return; }
  try {
    await navigator.clipboard.writeText(text);
    setStatus("feedbackStatus", "ok", "Copied to clipboard. (Feedback is not sent to a server in this build.)");
  } catch (err) {
    setStatus("feedbackStatus", "warn", "Clipboard unavailable. Please copy your text manually.");
  }
}

/* ------------------------------------------------------------------ *
 * AI Crop Insights
 *
 * The composer calls the backend's general crop Q&A endpoint
 * POST /insights/ask (see backend/app/routers/insights.py). The feature is
 * DISABLED by default and needs a server-side Groq key; when it is off the
 * endpoint returns a clear error envelope and this UI shows that message.
 * No answer is ever invented, no key is exposed to the browser, and prior
 * turns are NOT sent — the contract takes one question plus optional context
 * (predicted_label / confidence / family) and returns a bounded answer.
 * ------------------------------------------------------------------ */

const INSIGHT_TOPIC_LABELS = {
  disease_symptoms: "disease symptoms",
  disease_prevention: "disease prevention",
  crop_care: "crop care",
  irrigation: "irrigation",
  nutrients: "nutrients",
  prediction_interpretation: "interpreting a prediction",
};

const INSIGHT_KIND_LABELS = {
  question: "You asked",
  loading: "Thinking",
  answer: "AI answer",
  error: "Error",
  diagnosis: "Diagnosis reference",
  summary: "Backend summary",
  note: "Note",
};

function cropDisplayName(crop) {
  const found = state.crops.filter(function (c) { return c.crop === crop; })[0];
  return found ? found.display_name : (crop || "");
}

function insightTurnTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function addInsightTurn(kind, text, extra) {
  const session = state.insightSession;
  const opts = extra || {};
  session.turns.push({
    n: session.nextTurn,
    at: new Date(),
    kind: kind,
    text: text || "",
    meta: opts.meta || "",
    lines: opts.lines || null,
    disclaimer: opts.disclaimer || "",
  });
  session.nextTurn += 1;
  renderInsightSession();
  return session.turns[session.turns.length - 1];
}

function renderInsightSession() {
  const session = state.insightSession;
  const transcript = $("insightTranscript");
  if (!transcript) return;
  transcript.textContent = "";

  const meta = $("insightSessionMeta");
  if (meta) {
    meta.textContent = session.crop
      ? "Session crop: " + (session.cropDisplay || cropDisplayName(session.crop)) + " · " +
        session.turns.length + (session.turns.length === 1 ? " turn" : " turns") + " · session only"
      : "No active session yet. Choose a crop to begin.";
  }
  const hint = $("chartSessionHint");
  if (hint) {
    hint.textContent = session.crop ? "Session crop: " + (session.cropDisplay || cropDisplayName(session.crop)) : "";
  }

  if (!session.turns.length) {
    transcript.appendChild(el("p", { class: "muted-sm", text: "The conversation will appear here. Ask a question, or open this view from a diagnosis." }));
    return;
  }

  session.turns.forEach(function (turn) {
    const body = el("div", { class: "turn__body" });
    if (turn.lines && turn.lines.length) {
      if (turn.text) body.appendChild(el("p", { text: turn.text }));
      const ul = el("ul");
      turn.lines.forEach(function (line) { ul.appendChild(el("li", { text: line })); });
      body.appendChild(ul);
    } else {
      body.textContent = turn.text;
    }
    if (turn.disclaimer) {
      body.appendChild(el("div", { class: "turn__disclaimer", text: turn.disclaimer }));
    }
    const head = el("div", { class: "turn__head" }, [
      el("span", { class: "turn__badge", text: INSIGHT_KIND_LABELS[turn.kind] || turn.kind }),
      el("span", { class: "turn__meta", text: "Turn " + turn.n + " · " + insightTurnTime(turn.at) + (turn.meta ? " · " + turn.meta : "") }),
    ]);
    transcript.appendChild(el("div", { class: "turn turn--" + turn.kind }, [head, body]));
  });
  transcript.scrollTop = transcript.scrollHeight;
}

function setInsightCrop(crop, resetTurns) {
  const session = state.insightSession;
  const changed = session.crop && crop && session.crop !== crop;
  session.crop = crop || "";
  session.cropDisplay = crop ? cropDisplayName(crop) : "";
  if (resetTurns || changed) { session.turns = []; session.nextTurn = 1; }
  const select = $("insightCrop");
  if (select && crop) select.value = crop;
  const chartCrop = $("chartCrop");
  if (chartCrop && crop) chartCrop.value = crop;
  renderInsightSession();
  return changed;
}

function onInsightCropChange(event) {
  const crop = event.target.value;
  const session = state.insightSession;
  const changed = session.crop && session.crop !== crop;
  if (!crop || changed) cancelInsightRequest();
  if (!crop) {
    session.crop = "";
    session.cropDisplay = "";
    renderInsightSession();
    return;
  }
  const hadTurns = session.turns.length > 0;
  session.crop = crop;
  session.cropDisplay = cropDisplayName(crop);
  if (changed) { session.turns = []; session.nextTurn = 1; }
  const chartCrop = $("chartCrop");
  if (chartCrop) chartCrop.value = crop;
  renderInsightSession();
  if (changed) {
    addInsightTurn("note", "Started a new session for " + (session.cropDisplay || crop) + ". Earlier turns were cleared so answers from different crops are not mixed.");
  } else if (!hadTurns) {
    addInsightTurn("note", "Session started for " + (session.cropDisplay || crop) + ".");
  }
}

function insightContextFor(crop) {
  // Only attach the last prediction when it belongs to the session's crop, so
  // one crop's model output is never mixed into another crop's question.
  const r = state.lastResult;
  if (!r || !r.crop || r.crop !== crop) return null;
  const context = {};
  if (r.predicted_label) context.predicted_label = r.predicted_label;
  if (typeof r.confidence === "number" && isFinite(r.confidence)) context.confidence = r.confidence;
  if (r.model && r.model.family) context.family = r.model.family;
  return Object.keys(context).length ? context : null;
}

function setInsightAskBusy(busy) {
  const btn = $("insightAskBtn");
  if (!btn) return;
  btn.disabled = !!busy;
  btn.textContent = busy ? "Asking…" : "Ask";
}

function resolveInsightTurn(turn, kind, text, extra) {
  if (!turn) return;
  const opts = extra || {};
  turn.kind = kind;
  turn.text = text || "";
  turn.lines = opts.lines || null;
  turn.meta = opts.meta || "";
  turn.disclaimer = opts.disclaimer || "";
  renderInsightSession();
}

function removeInsightTurn(turn) {
  const turns = state.insightSession.turns;
  const index = turns.indexOf(turn);
  if (index === -1) return;
  turns.splice(index, 1);
  renderInsightSession();
}

function cancelInsightRequest() {
  if (state.insightController) {
    state.insightController.abort();
    state.insightController = null;
  }
  setInsightAskBusy(false);
}

async function submitInsight(event) {
  event.preventDefault();
  const crop = $("insightCrop").value;
  if (!crop) { setStatus("insightStatus", "warn", "Choose a crop for this session first."); return; }
  if (state.insightSession.crop !== crop) {
    setInsightCrop(crop, true);
    addInsightTurn("note", "Session started for " + (state.insightSession.cropDisplay || crop) + ".");
  }
  const topic = $("insightTopic").value;
  const question = $("insightQuestion").value.trim();
  if (!question) { setStatus("insightStatus", "warn", "Type a question to add it to the conversation."); return; }

  const topicLabel = INSIGHT_TOPIC_LABELS[topic] || topic;
  addInsightTurn("question", "Question: " + question, { meta: topicLabel });
  $("insightQuestion").value = "";

  // A new submit supersedes any in-flight question (only the latest answer is kept).
  if (state.insightController) state.insightController.abort();
  const controller = new AbortController();
  state.insightController = controller;

  const loading = addInsightTurn("loading", "Contacting the backend AI service…");
  setInsightAskBusy(true);
  setStatus("insightStatus", "info", "Waiting for the backend AI service…");

  try {
    const data = await apiRequest("/insights/ask", {
      method: "POST",
      signal: controller.signal,
      json: {
        crop: crop,
        question: question,
        topic: topic,
        context: insightContextFor(crop),
      },
    });
    const answer = data && typeof data.answer === "string" ? data.answer.trim() : "";
    if (!answer) {
      resolveInsightTurn(loading, "error", "The backend returned an empty answer. No answer was invented.");
      setStatus("insightStatus", "error", ERROR_MESSAGES.insights_empty_answer);
      return;
    }
    const meta = topicLabel + (data.model ? " · " + data.model : "");
    resolveInsightTurn(loading, "answer", answer, {
      meta: meta,
      disclaimer: data.disclaimer || "",
    });
    setStatus("insightStatus", "ok", "Answer received. AI-generated guidance — not a confirmed diagnosis.");
  } catch (err) {
    if (err && err.name === "AbortError") {
      removeInsightTurn(loading);
      return;
    }
    const message = describeError(err) || "The request failed.";
    const kind = (err instanceof ApiError && (err.code === "insights_disabled" || err.code === "insights_not_configured"))
      ? "warn"
      : "error";
    resolveInsightTurn(loading, "error", message);
    setStatus("insightStatus", kind, message);
  } finally {
    if (state.insightController === controller) {
      state.insightController = null;
      setInsightAskBusy(false);
    }
  }
}

function clearInsightSession() {
  cancelInsightRequest();
  state.insightSession = { crop: "", cropDisplay: "", turns: [], nextTurn: 1 };
  const select = $("insightCrop");
  if (select) select.value = "";
  renderInsightSession();
  setStatus("insightStatus", "ok", "Session cleared. The backend does not store this conversation.");
}

function exploreResultFromDiagnosis() {
  const result = state.lastResult;
  if (!result) { setStatus("diagnosisStatus", "warn", "Run a diagnosis first."); return; }
  const crop = result.crop || $("diagnosisCrop").value || "";
  const display = result.display_name || cropDisplayName(crop) || crop;
  setInsightCrop(crop, true);
  addInsightTurn("note", "Session started for " + display + ".");
  const pct = typeof result.confidence === "number" ? (result.confidence * 100).toFixed(1) + "%" : "—";
  addInsightTurn("diagnosis", "", {
    meta: display + " · model prediction",
    lines: [
      "Prediction: " + (result.predicted_label || "Unknown") + " (model confidence " + pct + ").",
      "This is a model prediction, not a confirmed diagnosis.",
      "Ask a question above, or use the charts and backend summary below to explore this crop.",
    ],
  });
  showView("insights");
  setStatus("insightStatus", "ok", "Diagnosis reference added. Ask a crop question above to use the AI crop Q&A.");
}

async function summarizeLastPrediction() {
  const out = $("insightSummaryOut");
  if (!state.lastResult) {
    out.hidden = true;
    setStatus("insightSummaryStatus", "warn", "No prediction in this session yet. Run a diagnosis in Crop Diagnosis first.");
    return;
  }
  setStatus("insightSummaryStatus", "info", "Generating summary…");
  const r = state.lastResult;
  try {
    const data = await apiRequest("/speech/response", {
      method: "POST",
      json: {
        crop: r.crop || null,
        display_name: r.display_name || null,
        predicted_label: r.predicted_label || null,
        confidence: typeof r.confidence === "number" ? r.confidence : null,
      },
    });
    const text = (data && data.text) ? data.text : "(empty summary)";
    out.hidden = false;
    out.textContent = text;
    const crop = r.crop || state.insightSession.crop;
    if (crop && state.insightSession.crop !== crop) setInsightCrop(crop, false);
    addInsightTurn("summary", text, { meta: (r.display_name || cropDisplayName(crop)) + " · /speech/response" });
    setStatus("insightSummaryStatus", "ok", "Backend summary of your last prediction. Model output, not a confirmed diagnosis.");
  } catch (err) {
    setStatus("insightSummaryStatus", "error", describeError(err));
  }
}

/* ---- Charts (all real backend data) ---- */

function onChartTypeChange() {
  const type = $("chartType").value;
  $("chartCropField").hidden = type !== "classsets";
  renderChart();
}

function showChartEmpty(message) {
  $("chartFigure").hidden = true;
  const empty = $("chartEmpty");
  empty.hidden = false;
  empty.textContent = "";
  empty.appendChild(el("p", { text: message }));
}

function renderBarChart(rows, opts) {
  const area = $("chartArea");
  area.textContent = "";
  const max = rows.reduce(function (acc, r) { return Math.max(acc, r.value); }, 0);
  const label = (opts && opts.title ? opts.title : "Chart") + ". " +
    rows.map(function (r) { return r.label + ": " + r.display; }).join("; ");
  const chart = el("div", { class: "bar-chart", role: "img", "aria-label": label });
  rows.forEach(function (row) {
    const width = max > 0 ? Math.max(0, Math.min(100, (row.value / max) * 100)) : 0;
    chart.appendChild(el("div", { class: "bar-row" }, [
      el("div", { class: "bar-row__label", text: row.label }),
      el("div", { class: "bar-row__track" }, [el("div", { class: "bar-row__fill", style: "width:" + width + "%;" })]),
      el("div", { class: "bar-row__value", text: row.display }),
    ]));
  });
  area.appendChild(chart);
}

function showChart(title, rows, caption) {
  $("chartEmpty").hidden = true;
  $("chartTitle").textContent = title;
  $("chartCaption").textContent = caption;
  renderBarChart(rows, { title: title });
  $("chartFigure").hidden = false;
}

async function renderChart() {
  const type = $("chartType").value;
  setStatus("chartStatus", null);

  if (type === "diagnosis") {
    const result = state.lastResult;
    if (!result || !Array.isArray(result.probabilities) || !result.probabilities.length) {
      showChartEmpty("No prediction available yet. Run a diagnosis, then return here.");
      return;
    }
    const rows = result.probabilities.map(function (p) {
      const pct = typeof p.probability === "number" ? p.probability * 100 : 0;
      return { label: p.label, value: pct, display: pct.toFixed(1) + "%" };
    });
    showChart(
      "Diagnosis probabilities — " + (result.display_name || result.crop),
      rows,
      "Model output scores from POST /predict (family: baseline). These are not a confirmed diagnosis."
    );
    return;
  }

  if (type === "models") {
    let payload = state.modelsPayload;
    if (!payload) {
      setStatus("chartStatus", "info", "Loading model inventory…");
      try {
        payload = await apiRequest("/models");
        state.modelsPayload = payload;
        setStatus("chartStatus", null);
      } catch (err) {
        showChartEmpty("Model inventory is unavailable.");
        setStatus("chartStatus", "error", describeError(err));
        return;
      }
    }
    const models = payload.models || [];
    if (!models.length) { showChartEmpty("The backend reported no models."); return; }
    const rows = models.map(function (m) {
      const name = (m.display_name || m.crop) + " (" + (m.family || "") + ")";
      return {
        label: name,
        value: typeof m.num_classes === "number" ? m.num_classes : 0,
        display: (typeof m.num_classes === "number" ? m.num_classes : "—") + " classes",
      };
    });
    showChart("Output classes per model", rows, "Number of classes each model predicts, from GET /models. A larger number is not a quality measure.");
    return;
  }

  if (type === "classsets") {
    const chartCrop = $("chartCrop");
    if (!chartCrop.value && state.insightSession.crop) chartCrop.value = state.insightSession.crop;
    const crop = chartCrop.value;
    if (!crop) { showChartEmpty("Select a crop to chart its class-set overlap."); return; }
    setStatus("chartStatus", "info", "Loading class sets…");
    try {
      const payload = await apiRequest("/comparison/" + encodeURIComponent(crop));
      setStatus("chartStatus", null);
      const counts = (payload.class_sets && payload.class_sets.counts) || {};
      const rows = [
        { label: "Shared labels", value: counts.shared || 0, display: String(counts.shared || 0) },
        { label: "Baseline-only labels", value: counts.baseline_only || 0, display: String(counts.baseline_only || 0) },
        { label: "Fine-tuned-only labels", value: counts.fine_tuned_only || 0, display: String(counts.fine_tuned_only || 0) },
      ];
      showChart(
        "Class-set overlap — " + (payload.display_name || crop),
        rows,
        "Counts of class labels shared or unique between models, from GET /comparison/" + crop + ". Label overlap only — not performance."
      );
    } catch (err) {
      showChartEmpty("Class-set data is unavailable for this crop.");
      setStatus("chartStatus", "error", describeError(err));
    }
    return;
  }
}

/* ------------------------------------------------------------------ *
 * Models (GET /models, GET /comparison/{crop})
 * ------------------------------------------------------------------ */

function yesNo(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "—";
}

function verificationBadgeClass(status) {
  if (status === "verified") return "badge-verified";
  if (status === "unverified") return "badge-unverified";
  return "badge-other";
}

function renderMetrics(entry) {
  const wrap = el("div");
  const metrics = entry.metrics || {};
  if (!metrics.available || !(metrics.items && metrics.items.length)) {
    wrap.appendChild(el("p", { class: "muted-sm", text: "No metrics reported for this model." }));
    return wrap;
  }
  const table = el("table", { class: "metric-table" });
  table.appendChild(el("thead", null, [
    el("tr", null, [
      el("th", { text: "Metric" }),
      el("th", { class: "num", text: "Value" }),
      el("th", { text: "Split" }),
      el("th", { text: "Provenance" }),
      el("th", { text: "Source" }),
    ]),
  ]));
  const body = el("tbody");
  metrics.items.forEach(function (item) {
    body.appendChild(el("tr", null, [
      el("td", { text: item.name }),
      el("td", { class: "num", text: typeof item.value === "number" ? String(item.value) : "—" }),
      el("td", { text: item.split || "—" }),
      el("td", { text: item.provenance || "—" }),
      el("td", { text: item.source || "—" }),
    ]));
  });
  table.appendChild(body);
  wrap.appendChild(table);
  if (metrics.test_samples != null) {
    wrap.appendChild(el("p", { class: "muted-sm", text: "Test samples: " + metrics.test_samples }));
  }
  if (metrics.disclaimer) {
    wrap.appendChild(el("div", { class: "disclaimer", text: metrics.disclaimer }));
  }
  if (metrics.notes && metrics.notes.length) {
    const ul = el("ul", { class: "notes" });
    metrics.notes.forEach(function (note) { ul.appendChild(el("li", { text: note })); });
    wrap.appendChild(ul);
  }
  return wrap;
}

const METRIC_FRIENDLY_NAMES = {
  test_accuracy: "Test accuracy",
  val_accuracy: "Validation accuracy",
  macro_f1: "Macro F1",
  test_macro_f1: "Test macro F1",
  test_weighted_f1: "Test weighted F1",
  val_loss: "Validation loss",
  test_loss: "Test loss",
};

function metricLabel(name) {
  return METRIC_FRIENDLY_NAMES[name] || name || "metric";
}

// A metric is drawn as a 0–100%bar only when it is a genuine 0–1 ratio that is
// not a loss. Everything else is shown as a raw number with no bar.
function metricIsFraction(item) {
  return typeof item.value === "number" && isFinite(item.value) &&
    item.value >= 0 && item.value <= 1 && !/loss/i.test(item.name || "");
}

function formatMetricValue(item) {
  if (typeof item.value !== "number" || !isFinite(item.value)) return "—";
  if (metricIsFraction(item)) return (item.value * 100).toFixed(1) + "%";
  return String(item.value);
}

function provenanceBadgeClass(provenance) {
  if (provenance === "provider-reported") return "badge-other";
  if (provenance === "training-reported") return "badge-local";
  return "badge-unverified";
}

function renderMetricBar(item) {
  const fraction = metricIsFraction(item);
  const split = item.split || "unknown";
  const provenance = item.provenance || "unknown";
  const row = el("div", { class: "metric-bar" + (fraction ? "" : " metric-bar--raw") });
  row.appendChild(el("div", { class: "metric-bar__head" }, [
    el("span", { class: "metric-bar__name", text: metricLabel(item.name) }),
    el("span", { class: "metric-bar__value", text: formatMetricValue(item) }),
  ]));
  row.appendChild(el("div", { class: "metric-bar__meta" }, [
    el("span", { class: "badge " + (split === "unknown" ? "badge-unverified" : "badge-aux"), text: split === "unknown" ? "split unknown" : split + " split" }),
    el("span", { class: "badge " + provenanceBadgeClass(provenance), text: provenance }),
  ]));
  if (fraction) {
    const pct = Math.max(0, Math.min(100, item.value * 100));
    row.appendChild(el("div", {
      class: "metric-bar__track",
      role: "img",
      "aria-label": metricLabel(item.name) + ": " + formatMetricValue(item) + ", " + split + " split, " + provenance,
    }, [el("div", { class: "metric-bar__fill", style: "width:" + pct + "%;" })]));
  } else {
    row.appendChild(el("p", { class: "muted-sm", text: "Reported as a raw value (not shown as a bar)." }));
  }
  return row;
}

function renderMetricCards(entry) {
  const wrap = el("div", { class: "metric-cards" });
  const metrics = entry.metrics || {};
  if (!metrics.available || !(metrics.items && metrics.items.length)) {
    wrap.appendChild(el("p", { class: "muted-sm", text: "No metrics reported for this model." }));
    return wrap;
  }
  metrics.items.forEach(function (item) { wrap.appendChild(renderMetricBar(item)); });
  return wrap;
}

function findMetric(entry, name, split) {
  const items = (entry && entry.metrics && entry.metrics.items) || [];
  for (let i = 0; i < items.length; i++) {
    if (items[i].name === name && (split === undefined || items[i].split === split)) return items[i];
  }
  return null;
}

function renderModelPanel(entry) {
  const panel = el("div", { class: "model-panel" });
  panel.appendChild(el("div", { class: "model-panel__head" }, [
    el("span", { class: "model-panel__role", text: entry.display_role || entry.family || "Model" }),
    el("div", { class: "model-panel__badges" }, [
      el("span", { class: "badge " + (entry.live_inference_enabled ? "badge-official" : "badge-unverified"), text: entry.live_inference_enabled ? "Live inference enabled" : "Live inference disabled" }),
      el("span", { class: "badge " + verificationBadgeClass(entry.verification_status), text: entry.verification_status || "unknown" }),
    ]),
  ]));
  const bits = [];
  if (entry.architecture) bits.push(entry.architecture);
  if (entry.num_classes != null) bits.push(entry.num_classes + " classes");
  if (Array.isArray(entry.input_size) && entry.input_size.length) bits.push(entry.input_size.join(" × "));
  panel.appendChild(el("p", { class: "model-panel__meta", text: bits.length ? bits.join(" · ") : "No architecture metadata reported for this model." }));
  panel.appendChild(renderMetricCards(entry));
  if (!entry.live_inference_enabled && entry.live_inference_blocked_reason) {
    panel.appendChild(el("p", { class: "muted-sm", text: "Not serving predictions: " + entry.live_inference_blocked_reason }));
  }
  return panel;
}

function renderMissingPanel(roleLabel) {
  const panel = el("div", { class: "model-panel model-panel--empty" });
  panel.appendChild(el("div", { class: "model-panel__head" }, [
    el("span", { class: "model-panel__role", text: roleLabel }),
  ]));
  panel.appendChild(el("p", { class: "muted-sm", text: "Not reported for this crop — no checkpoint was found on the backend." }));
  return panel;
}

// Explicitly states whether baseline vs fine-tuned values are comparable. They
// never are when class sets or evaluation protocols differ, and even matching
// metric names are NOT a quality ranking.
function comparisonNote(baseline, ft) {
  if (!ft) {
    return "Only a baseline model is reported for this crop, so there is no fine-tuned checkpoint to compare against.";
  }
  const bTest = findMetric(baseline, "test_accuracy", "test");
  const fTest = findMetric(ft, "test_accuracy", "test");
  if (bTest && fTest) {
    const bClasses = baseline.num_classes != null ? baseline.num_classes : "?";
    const fClasses = ft.num_classes != null ? ft.num_classes : "?";
    return "Both models happen to report test_accuracy, but on different class sets (" + bClasses + " vs " + fClasses +
      " classes) and under different evaluation protocols (provider-reported vs training-reported). The panels sit side by side for inspection only and must NOT be read as a head-to-head ranking.";
  }
  return "The baseline and fine-tuned models do not report the same named metric on the same split, so their metric values are not directly comparable.";
}

function renderEntryDetailsBody(title, entry) {
  const wrap = el("div", { class: "tech-details__body" });
  wrap.appendChild(el("h4", { class: "tech-details__subhead", text: title }));

  const kv = el("div", { class: "kv-grid" });
  function pair(key, value) {
    kv.appendChild(el("div", { class: "kv" }, [
      el("span", { class: "kv__k", text: key }),
      el("span", { class: "kv__v", text: value }),
    ]));
  }
  pair("Family", entry.family || "—");
  pair("Architecture", entry.architecture || "—");
  pair("Implementation", entry.implementation || "—");
  pair("Classes", entry.num_classes != null ? String(entry.num_classes) : "—");
  pair("Input size", Array.isArray(entry.input_size) && entry.input_size.length ? entry.input_size.join(" × ") : "—");
  pair("Checkpoint present", yesNo(entry.checkpoint_present));
  pair("Checkpoint loads", yesNo(entry.checkpoint_loads));
  pair("Preprocessing verified", yesNo(entry.preprocessing_verified));
  pair("Verification status", entry.verification_status || "—");
  pair("Live inference", entry.live_inference_enabled ? "enabled" : "disabled");
  if (entry.live_inference_blocked_reason) pair("Blocked reason", entry.live_inference_blocked_reason);
  if (entry.training_info && entry.training_info.best_epoch != null) pair("Best epoch", String(entry.training_info.best_epoch));
  wrap.appendChild(kv);

  wrap.appendChild(renderMetrics(entry));

  if (entry.labels && entry.labels.length) {
    wrap.appendChild(el("p", { class: "class-set__label", text: "Class labels (" + entry.labels.length + ")" }));
    const chips = el("div", { class: "chips" });
    entry.labels.forEach(function (label) { chips.appendChild(el("span", { class: "chip", text: label })); });
    wrap.appendChild(chips);
  }
  return wrap;
}

function renderCropOverview(display, baseline, ft) {
  const card = el("article", { class: "card crop-overview" });

  card.appendChild(el("div", { class: "card__head" }, [
    el("h3", { class: "card__title", text: display }),
    el("div", { class: "crop-overview__badges" }, [
      baseline && baseline.live_inference_enabled
        ? el("span", { class: "badge badge-official", text: "Baseline serving" })
        : el("span", { class: "badge badge-unverified", text: "Baseline not serving" }),
      ft
        ? el("span", { class: "badge badge-aux", text: "Fine-tuned present" })
        : el("span", { class: "badge badge-unverified", text: "No fine-tuned model" }),
    ]),
  ]));

  const present = [baseline, ft].filter(Boolean);
  const liveNow = present.filter(function (e) { return e.live_inference_enabled; }).length;
  const reported = present.filter(function (e) { return e.metrics && e.metrics.available; }).length;
  const tiles = el("div", { class: "stat-row" });
  function tile(value, label) {
    tiles.appendChild(el("div", { class: "stat-tile" }, [
      el("span", { class: "stat-tile__value", text: value }),
      el("span", { class: "stat-tile__label", text: label }),
    ]));
  }
  tile(String(present.length), "models reported");
  tile(String(liveNow), "serving live");
  tile(String(reported), "with metrics");
  card.appendChild(tiles);

  const compare = el("div", { class: "model-compare" });
  compare.appendChild(baseline ? renderModelPanel(baseline) : renderMissingPanel("Baseline model"));
  compare.appendChild(ft ? renderModelPanel(ft) : renderMissingPanel("Fine-tuned checkpoint"));
  card.appendChild(compare);

  card.appendChild(el("div", { class: "disclaimer", text: "Comparison note: " + comparisonNote(baseline, ft) }));

  const details = el("details", { class: "tech-details" });
  details.appendChild(el("summary", { class: "tech-details__summary", text: "Technical details & provenance" }));
  if (baseline) details.appendChild(renderEntryDetailsBody("Baseline model", baseline));
  if (ft) details.appendChild(renderEntryDetailsBody("Fine-tuned checkpoint", ft));
  card.appendChild(details);

  return card;
}

function renderModels(payload) {
  const list = $("modelsList");
  list.textContent = "";
  const models = (payload && payload.models) || [];
  if (!models.length) {
    list.appendChild(el("p", { class: "muted-sm", text: "No models reported by the backend." }));
    return;
  }

  const order = [];
  const byCrop = {};
  models.forEach(function (entry) {
    const key = entry.crop || entry.display_name || "unknown";
    if (!byCrop[key]) {
      byCrop[key] = { display: entry.display_name || key, models: {} };
      order.push(key);
    }
    byCrop[key].models[entry.family || "baseline"] = entry;
  });

  order.forEach(function (key) {
    const group = byCrop[key];
    list.appendChild(renderCropOverview(group.display, group.models.baseline || null, group.models.fine_tuned || null));
  });
}

async function loadModels() {
  setStatus("modelsStatus", "info", "Loading model inventory…");
  try {
    const payload = await apiRequest("/models");
    state.modelsPayload = payload;
    setStatus("modelsStatus", null);
    renderModels(payload);
  } catch (err) {
    state.modelsPayload = null;
    $("modelsList").textContent = "";
    setStatus("modelsStatus", "error", describeError(err));
  }
}

async function loadComparison(crop) {
  const body = $("comparisonBody");
  body.textContent = "";
  if (!crop) { setStatus("comparisonStatus", null); return; }
  setStatus("comparisonStatus", "info", "Loading class sets…");
  try {
    const payload = await apiRequest("/comparison/" + encodeURIComponent(crop));
    setStatus("comparisonStatus", null);
    renderComparison(payload);
  } catch (err) {
    setStatus("comparisonStatus", "error", describeError(err));
  }
}

function renderComparison(payload) {
  const body = $("comparisonBody");
  body.textContent = "";
  const counts = (payload.class_sets && payload.class_sets.counts) || {};

  const grid = el("div", { class: "kv-grid" });
  function kv(key, value) {
    grid.appendChild(el("div", { class: "kv" }, [
      el("span", { class: "kv__k", text: key }),
      el("span", { class: "kv__v", text: value }),
    ]));
  }
  kv("Baseline classes", counts.baseline != null ? String(counts.baseline) : "—");
  kv("Fine-tuned classes", counts.fine_tuned != null ? String(counts.fine_tuned) : "—");
  kv("Shared", counts.shared != null ? String(counts.shared) : "—");
  kv("Baseline only", counts.baseline_only != null ? String(counts.baseline_only) : "—");
  kv("Fine-tuned only", counts.fine_tuned_only != null ? String(counts.fine_tuned_only) : "—");
  kv("Direct alignment", yesNo(payload.class_sets && payload.class_sets.direct_class_alignment_possible));
  body.appendChild(grid);

  function chipList(label, items, cls, getText) {
    if (!items || !items.length) return;
    body.appendChild(el("p", { class: "class-set__label", text: label }));
    const chips = el("div", { class: "chips" });
    items.forEach(function (item) {
      chips.appendChild(el("span", { class: "chip " + (cls || ""), text: getText(item) }));
    });
    body.appendChild(chips);
  }
  const sets = payload.class_sets || {};
  chipList("Shared labels", sets.shared, "", function (s) { return s.baseline_label + " / " + s.fine_tuned_label; });
  chipList("Baseline-only labels", sets.baseline_only, "chip-baseline-only", function (s) { return s.label; });
  chipList("Fine-tuned-only labels", sets.fine_tuned_only, "chip-fine-only", function (s) { return s.label; });

  if (payload.model_comparison) {
    const mc = payload.model_comparison;
    body.appendChild(el("p", { class: "muted-sm", text: "Live prediction comparison possible: " + yesNo(mc.live_prediction_comparison_possible) + " (endpoint " + (mc.live_prediction_endpoint || "/predict") + ")" }));
    if (mc.live_prediction_comparison_blocked_reason) {
      body.appendChild(el("p", { class: "muted-sm", text: "Blocked: " + mc.live_prediction_comparison_blocked_reason }));
    }
    if (mc.note) {
      body.appendChild(el("p", { class: "muted-sm", text: mc.note }));
    }
  }

  if (payload.disclaimer) {
    body.appendChild(el("div", { class: "disclaimer", text: payload.disclaimer }));
  }
}

/* ------------------------------------------------------------------ *
 * Settings display
 * ------------------------------------------------------------------ */

function renderSettings() {
  $("apiBaseDisplay").textContent = API.base;
  $("apiBaseSource").textContent = API.source;
}

/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */

function bindEvents() {
  qsa(".nav-item").forEach(function (btn) {
    btn.addEventListener("click", function () { showView(btn.dataset.view); });
  });
  qsa("[data-goto]").forEach(function (btn) {
    btn.addEventListener("click", function () { showView(btn.dataset.goto); });
  });

  $("menuToggle").addEventListener("click", function () {
    if ($("sidebar").classList.contains("is-open")) closeSidebar();
    else openSidebar();
  });
  $("overlay").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeSidebar();
  });

  $("diagnosisForm").addEventListener("submit", submitDiagnosis);
  $("resetDiagnosisBtn").addEventListener("click", resetDiagnosis);
  $("takePhotoBtn").addEventListener("click", function () { $("leafCamera").click(); });
  $("uploadPhotoBtn").addEventListener("click", function () { $("leafUpload").click(); });
  $("leafCamera").addEventListener("change", onFileSelected);
  $("leafUpload").addEventListener("change", onFileSelected);
  $("diagnosisCrop").addEventListener("change", function () {
    if (state.selectedLeafFile && !$("previewWrap").hidden) renderPreviewMeta(state.selectedLeafFile);
  });

  $("recordBtn").addEventListener("click", startRecording);
  $("stopRecordBtn").addEventListener("click", stopRecording);
  $("transcribeFile").addEventListener("change", onTranscribeFileSelected);

  $("reloadModelsBtn").addEventListener("click", loadModels);
  $("comparisonCrop").addEventListener("change", function (event) { loadComparison(event.target.value); });

  $("schemeForm").addEventListener("submit", submitSchemeSearch);
  $("schemeClearBtn").addEventListener("click", clearSchemeForm);
  qsa('input[name="schemeMode"]').forEach(function (input) {
    input.addEventListener("change", onSchemeModeChange);
  });
  setSchemeMode(currentSchemeMode());

  $("cropForm").addEventListener("submit", function (event) {
    event.preventDefault();
    const name = $("cropNameInput").value.trim();
    if (!name) return;
    state.sessionCrops.push({ name: name, at: new Date() });
    $("cropNameInput").value = "";
    renderMyCrops();
  });

  $("clearHistoryBtn").addEventListener("click", function () {
    state.sessionHistory = [];
    renderHistory();
    renderDashboardSession();
  });
  $("clearSessionBtn").addEventListener("click", clearSession);

  $("feedbackBtn").addEventListener("click", copyFeedback);

  $("voiceToggle").addEventListener("change", function (event) { setVoice(event.target.checked); });
  $("stopSpeechBtn").addEventListener("click", stopSpeech);
  $("readPageBtn").addEventListener("click", startBrowserRead);
  $("ttsPauseBtn").addEventListener("click", togglePauseRead);
  $("ttsStopBtn").addEventListener("click", stopBrowserSpeech);
  $("recheckHealthBtn").addEventListener("click", function () { loadHealth(); loadCrops(); });

  $("insightForm").addEventListener("submit", submitInsight);
  $("insightCrop").addEventListener("change", onInsightCropChange);
  $("insightClearBtn").addEventListener("click", clearInsightSession);
  $("insightSummarizeBtn").addEventListener("click", summarizeLastPrediction);
  $("chartType").addEventListener("change", onChartTypeChange);
  $("chartCrop").addEventListener("change", renderChart);
  $("chartRenderBtn").addEventListener("click", renderChart);
}

function init() {
  renderSettings();

  let storedVoice = "1";
  try { storedVoice = localStorage.getItem("ai4seva.voice") || "1"; } catch (e) { /* ignore */ }
  setVoice(storedVoice === "1");

  bindEvents();
  initTTSVoices();
  initSectionReaders();
  showView("dashboard");
  renderMyCrops();
  renderHistory();
  renderDashboardSession();
  renderInsightSession();

  loadHealth();
  loadCrops();
  loadModels();
  onChartTypeChange();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
