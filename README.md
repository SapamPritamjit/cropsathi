# AI4Seva

AI-powered crop leaf disease detection platform. Supports **Maize, Rice, Potato,
and Tomato**, with baseline and fine-tuned model comparison, an image-prediction
API, and voice accessibility (text-to-speech, speech-to-text, and a
farmer-friendly prediction-to-speech response builder).

> Status: **backend implemented** — the health, crops, prediction,
> model-inventory, comparison, and speech endpoints below are registered and
> covered by tests. The web frontend/dashboard is **not yet built**. Fine-tuned
> predictions are implemented and tested but **disabled in the public API** until
> the training preprocessing is confirmed (see below).

## Repository layout

```text
AI4Seva Hackathon/
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── main.py              # app factory, CORS, error handlers
│   │   ├── config.py            # settings + artifact path discovery
│   │   ├── errors.py            # ApiError -> JSON error envelope
│   │   ├── imaging.py           # image decode/validation (no torch)
│   │   ├── preprocessing.py     # transform from verified metadata
│   │   ├── inference.py         # adapter interface + model registry + predict
│   │   ├── model_metadata.py    # reads JSON metadata only (no checkpoints)
│   │   ├── model_verification.py# checkpoint load/verification (no inference)
│   │   ├── comparison.py        # model inventory + comparison (no inference)
│   │   ├── schemas.py           # Pydantic models
│   │   ├── speech/              # tts.py, stt.py, text.py (Groq-backed voice)
│   │   └── routers/             # health.py, crops.py, predict.py,
│   │                            # comparison.py, speech.py
│   ├── scripts/verify_models.py # offline artifact verification report
│   ├── tests/                   # pytest suite
│   ├── requirements.txt
│   └── .env.example
├── <crop>-disease-convnext/     # baseline ConvNeXt-Small artifacts (weights git-ignored)
├── fine-tune-model/             # fine-tuned EfficientNet-B0 artifacts (weights git-ignored)
├── SpeechToText.py              # provided legacy voice asset (not used by the API)
└── TextToSpeech.py              # provided legacy voice asset (not used by the API)
```

Model weights and dataset manifests are **not** committed (see `.gitignore`).

## Backend setup

Windows PowerShell:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs> (schema at `/openapi.json`).

### Environment

Configuration is read from the project-root `.env` (copy `backend/.env.example`).
Secrets stay **server-side only** and are never returned by any endpoint. `.env`
is git-ignored; only `.env.example` is committed.

Key variables:

- `GROQ_API_KEY` — Groq API key (`GROQ-API-KEY` also accepted). Required for live
  text-to-speech and speech-to-text; never logged or exposed.
- `CORS_ALLOWED_ORIGINS` — comma-separated list of exact browser origins allowed
  to call the API, e.g.
  `CORS_ALLOWED_ORIGINS=http://localhost:5173,https://your-app.lovable.app`.
  **Empty by default (deny all cross-origin browser requests)**; no wildcard and
  no credentials. Set this before connecting a browser frontend.
- `GROQ_TTS_*` — text-to-speech endpoint/model/voice/format/limits.
- `GROQ_STT_*` and `STT_MAX_UPLOAD_BYTES` — speech-to-text endpoint/model/format/
  timeout and upload ceiling.

Defaults for every variable are documented in `backend/.env.example` and match
`backend/app/config.py`.

### Endpoints

| Method | Path                     | Description                                                        |
|--------|--------------------------|--------------------------------------------------------------------|
| GET    | `/health`                | Liveness + safe service information.                               |
| GET    | `/crops`                 | Supported crops + baseline / fine-tuned artifact presence.         |
| POST   | `/predict`               | `multipart/form-data` (`crop`, `file`, optional `family`) → class. |
| GET    | `/models`                | Inventory of all model artifacts + verification/load status.       |
| GET    | `/comparison/{crop}`     | Baseline vs fine-tuned metadata, class-set overlap, metrics.      |
| POST   | `/speech/speak`          | Synthesize text to **WAV** audio (Groq TTS).                       |
| POST   | `/speech/transcribe`     | Transcribe an uploaded audio clip to text (Groq STT).              |
| POST   | `/speech/response`       | Build farmer-friendly speech text from a prediction.               |

Errors use a consistent envelope:

```json
{ "error": { "code": "404", "message": "Not Found", "details": null } }
```

### Prediction

`POST /predict` uses the crop's verified **baseline** ConvNeXt-Small model by
default and returns the predicted human-readable label, confidence, full
probability distribution, and verified model metadata. Confidence is the model's
softmax probability, **not** a guarantee of diagnosis.

`family=baseline` (default) | `fine_tuned`. Fine-tuned EfficientNet-B0 inference
(maize, rice) is implemented but **disabled** until the training preprocessing is
confirmed; requesting it returns `503 fine_tuned_preprocessing_unverified`
unless `ENABLE_FINE_TUNED_PREDICTIONS=1` is set. The supplied checkpoints record
architecture/classes/input-size but **not** the training resize or normalization,
so an ImageNet-style transform is only *assumed* (`preprocessing.py`,
`EFFICIENTNET_ASSUMED_CFG`).

### Model inventory & comparison

`GET /models` lists every artifact (6 total: 4 baselines + maize & rice
fine-tuned) with architecture, class count, labels, checkpoint presence/load
status, and whether live inference is enabled. `GET /comparison/{crop}` returns
the baseline entry, the fine-tuned entry when one exists, the class-set overlap
(shared / baseline-only / fine-tuned-only), and metric provenance. Metrics are
**never** compared across different class sets, and all numbers are labelled by
source: `"Provider-reported; not locally reproduced."` (baseline README) or
`"Training-reported; not independently reproduced."` (fine-tuned checkpoint).
Live baseline-vs-fine-tuned prediction comparison is blocked while fine-tuned
preprocessing is unverified.

### Voice

- `POST /speech/speak` sends text to Groq's Orpheus model and returns raw WAV
  bytes. **Input is capped at 200 characters** (over-limit → `413
  tts_text_too_long`), and **only WAV output** is supported. Requires
  `GROQ_API_KEY`; missing key → `503 tts_not_configured`.
- `POST /speech/transcribe` accepts an audio upload (`wav/mp3/m4a/ogg/webm/flac`,
  bounded by `STT_MAX_UPLOAD_BYTES`) and returns the transcript. The optional
  `language` hint is ISO-639-1; **which languages are actually supported depends
  on the configured transcription service** (Groq Whisper), not on this API.
- `POST /speech/response` converts a prediction (`crop`/`display_name`,
  `predicted_label`, optional `confidence`) into concise, uncertainty-preserving
  farmer-friendly text. It is a pure transformation — no inference or network
  calls.

## Tests

```powershell
cd backend
python -m pytest
```
