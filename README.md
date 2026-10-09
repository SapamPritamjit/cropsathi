# AI4Seva

AI-powered crop leaf disease detection platform. Supports **Maize, Rice, Potato,
and Tomato**, with baseline and fine-tuned model comparison, an image-prediction
API, and voice accessibility (text-to-speech, speech-to-text, and a
farmer-friendly prediction-to-speech response builder).

> Status: **backend implemented** — the health, crops, prediction,
> model-inventory, comparison, speech and farmer-scheme-search endpoints below
> are registered and covered by tests. The web frontend/dashboard is **not yet
> built**. Fine-tuned
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
│   │   ├── schemes/            # Farmer Scheme Finder: provider, sources, service
│   │   └── routers/             # health.py, crops.py, predict.py,
│   │                            # comparison.py, speech.py, schemes.py
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
- `SCHEME_SEARCH_ENABLED` — master switch for `GET /schemes/search` (**default
  `false`** → `503 scheme_search_disabled`). `SCHEME_SEARCH_TIMEOUT_SECONDS`
  (default 8) and `SCHEME_SEARCH_MAX_RESULTS` (default 20) bound the provider
  call; `SCHEME_SEARCH_REGION` (default `in-en`, India/English) is the DDGS
  region. `SCHEME_OFFICIAL_ORIGINS` is the explicit official-source allow-list
  (**empty by default**, so no host is marked verified). Evidence-checked hosts
  (e.g. `agrimanipur.mn.gov.in`, `manipur.gov.in`, `agriwelfare.gov.in`) are
  suggested in `backend/.env.example`, but operators **must review and set the
  production allow-list themselves**; add exact hosts only, never broad
  suffixes such as `gov.in`.
- `SCHEME_CATALOGUE_ENABLED` — merge a small, hand-curated set of official
  information resources into `GET /schemes/search` results (**default `false`**).
  Curated entries are merged only **after a successful live search** (never as a
  fallback for provider failures/timeouts) and are only marked verified when
  their host is accepted by `SCHEME_OFFICIAL_ORIGINS`: the catalogue is **not** a
  second allow-list. Seeded entries carry `scheme_status="unknown"` — domain
  identity was reviewed, but scheme activity, eligibility and availability were
  **not** verified.

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
| GET    | `/schemes/search`        | Search official farmer schemes (Manipur-focused); **disabled by default**. |

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

### Farmer scheme search

`GET /schemes/search` searches official farmer schemes with Manipur context. It
is **disabled by default** (`SCHEME_SEARCH_ENABLED=false`) and returns `503
scheme_search_disabled` until enabled. At least one of `q`, `crop` or `need`
must be supplied (a `district`/`language` alone does not trigger a search).

Query parameters:

| Name       | Required | Notes                                                             |
|------------|----------|-------------------------------------------------------------------|
| `q`        | one of   | Free text, trimmed, max 200 chars.                                |
| `crop`     | one of   | `maize` \| `rice` \| `potato` \| `tomato`.                        |
| `need`     | one of   | `seeds`,`fertilizer`,`credit`,`insurance`,`irrigation`,`machinery`, `market`,`training`,`pest_disease`,`general`. |
| `district` | no       | Manipur district name (bounded free text; no fixed list verified). |
| `limit`    | no       | 1–20 (default 5); also capped by `SCHEME_SEARCH_MAX_RESULTS`.     |
| `language` | no       | ISO-639-1 code (default `en`); this API does **not** claim TTS support for every language. |

Example request:

```text
GET /schemes/search?crop=rice&district=Imphal%20East&need=insurance&language=en
```

Example response (shape only):

```json
{
  "query": { "q": null, "crop": "rice", "district": "Imphal East", "need": "insurance" },
  "results": [
    {
      "title": "Example result",
      "url": "https://example.gov.in/scheme",
      "snippet": "Search-provider summary...",
      "source_domain": "example.gov.in",
      "source_type": "official_goi",
      "verified": true
    }
  ],
  "result_count": 1,
  "official_count": 1,
  "retrieved_at": "2026-01-01T00:00:00+00:00",
  "disclaimer": "Search results are informational. Confirm scheme availability and eligibility with the responsible official department.",
  "language": "en"
}
```

`source_type` is `official_manipur` / `official_goi` / `other`, and `verified` is
true **only** for hosts on the explicit `SCHEME_OFFICIAL_ORIGINS` allow-list
(empty by default). Results are ordered official-first (verified Manipur, then
verified Government of India, then unverified), and within each tier by a simple
deterministic relevance score derived from the Manipur context, the selected
`crop`/`need` and the query terms. Duplicate results are collapsed using
conservative URL normalization (host casing, fragments, tracking parameters and
trailing slashes; meaningful query parameters are preserved). A successful search
that finds nothing returns `200` with empty results; only genuine provider
failures (`502`) and timeouts (`504`) are errors. Result URLs are never fetched,
and **a search hit is not proof of eligibility, an open application, or a
guaranteed benefit**.

## Tests

```powershell
cd backend
python -m pytest
```
