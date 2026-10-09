# AI4Seva Frontend (`frontend_app`)

Desktop-first, plain HTML/CSS/JavaScript frontend for the AI4Seva crop-disease
screening and farmer-scheme services. No framework, no bundler, no build step.

## Folder contents

| File | Purpose |
| --- | --- |
| `index.html` | Single-page markup: sidebar navigation and all sections. |
| `styles.css` | All styling (desktop-first, responsive down to narrow screens). |
| `app.js` | Application logic: centralized API helper, views, diagnosis, camera/upload, speech (page + per-section reading), schemes, models, continuous AI Crop Insights, charts. |
| `README.md` | This document. |

No local images, fonts, or third-party JS. One remote Unsplash hero image is
referenced from CSS as decoration; if it fails to load the hero falls back to a
solid green panel.

## 1. Start the backend

```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Confirm with <http://127.0.0.1:8000/health>.

## 2. Serve the frontend over HTTP

Do **not** open `index.html` from disk (`file://`): that sends `Origin: null`,
which the backend CORS policy rejects by default. Serve the folder statically:

```powershell
cd frontend_app
python -m http.server 5500
```

Open <http://localhost:5500/>. The frontend origin is `http://localhost:5500`.

## 3. Configure the backend base URL

Resolved once, in order:

1. `window.CROPSATHI_API_BASE_URL` if set before `app.js` loads.
2. Otherwise the documented default `http://127.0.0.1:8000`.

```html
<script>window.CROPSATHI_API_BASE_URL = "http://192.168.1.20:8000";</script>
<script src="app.js"></script>
```

All requests go through the single `apiRequest` helper; the URL is not
hard-coded elsewhere. The resolved value is shown in **Settings → Backend
connection**.

## 4. Backend environment settings the operator must configure

Set in the backend environment (e.g. `backend/.env`), never in the frontend.

- **`CORS_ALLOWED_ORIGINS`** — must include the exact frontend origin, e.g.
  `CORS_ALLOWED_ORIGINS=http://localhost:5500`. Default empty denies all
  cross-origin browser requests.
- **`SCHEME_SEARCH_ENABLED=true`** and provider settings — required for live
  scheme search; otherwise the endpoint returns `scheme_search_disabled` /
  `scheme_search_not_configured` and the UI says so.
- **`GROQ-API-KEY`** (or `GROQ_API_KEY`) — required for `/speech/speak`,
  `/speech/transcribe` and `/speech/response`; without it they return
  `*_not_configured` (503). Never expose this key to the frontend.
- **Image/audio limits** — `IMAGE_MAX_BYTES` (default 8 MB), `STT_MAX_UPLOAD_BYTES`
  (default 25 MB). The frontend pre-checks 8 MB / 25 MB and common types for UX,
  but the backend is authoritative.

## 5. Endpoint-to-UI coverage matrix

Legend: **Complete** = usable and faithful to the backend; **Backend-limited** =
backend cannot support the UI need.

| Backend capability | UI location | Status |
| --- | --- | --- |
| `GET /health` | Sidebar pill | Complete |
| `GET /crops` | Crop selects (Diagnosis, Schemes, Models, Insights, Charts) | Complete |
| `GET /models` | **Models**; **Insights → charts** | Complete |
| `GET /comparison/{crop}` | **Models**; **Insights → charts** | Complete |
| `POST /predict` | **Crop Diagnosis**; feeds diagnosis chart | Complete |
| `POST /speech/response` | Diagnosis spoken summary; **Insights → Interpret a prediction** | Complete |
| `POST /speech/speak` | Spoken-summary **Listen**/**Stop**; browser fallback | Complete |
| `POST /speech/transcribe` | **Crop Diagnosis → Voice note transcription** | Complete |
| `GET /schemes/search` | **Scheme Finder** | Complete |
| `POST /insights/ask` | **Insights → composer** (Ask) | Complete when enabled — off by default, needs a server Groq key (see §7) |
| Error envelope | All views via `describeError` | Complete |

## 6. Crop Diagnosis: camera vs. upload

Two explicit input paths feed the **same** validation and `POST /predict` call:

- **Take a photo** — opens a hidden `<input type="file" accept="image/*"
  capture="environment">`. Whether a camera actually opens depends on the device
  and browser; on many desktops it falls back to a file picker. This is not
  guaranteed by the page.
- **Upload an image** — a normal file picker with no `capture` attribute.

Behaviour:

- Both handlers call one `onFileSelected`, which validates type and size, then
  previews the image with an object URL (`URL.createObjectURL`).
- The preview is a two-column image + file-info layout on desktop and stacks on
  narrow screens. It shows the selected **crop**, a clean (path-stripped,
  shortened) **filename** with the full name in a tooltip, and the **file size**
  — never a path, object URL or encoded data. Long filenames wrap
  (`overflow-wrap: anywhere`) and cannot push controls outside the card.
- Selecting an image through one path clears the other input so the two never
  disagree; the tracked file lives in `state.selectedLeafFile`.
- The old object URL is revoked whenever the preview is replaced or the
  diagnosis is reset/left, and `img.onerror` handles a broken/unreadable
  preview by clearing the selection and showing an error.
- Empty selections/ cancellation are ignored; unsupported types, oversized
  files, and preview errors show a visible message.
- The backend still enforces its real size/format constraints.

Each analysis is tagged with a request id. While one is in flight the Analyze
button and the crop/photo controls are disabled, the previous result is cleared
and clearly marked as replaced, and any response belonging to a superseded
request is ignored — so a slow earlier response can never overwrite a newer
prediction, and an error or incomplete payload is never shown as a result.

**Low-confidence candidates.** One centralized constant sets the display rule:
`LOW_CONFIDENCE_THRESHOLD` (default **0.75** = 75 %). When the top model
confidence is **below 75 %**, the result card presents the **top three classes
ranked by their actual returned probabilities** (`CANDIDATE_CLASS_COUNT`), with
the leading candidate and its score called out, the alternatives below in a
ranked list, and a plain-language note that these are the highest-scoring
classes, *not confirmed diagnoses*. At **75 % or higher** the card keeps the
existing single-primary-prediction layout. In both cases the full class
distribution, speaker controls, `disclaimer`, and
**Continue exploring this crop** remain. The technical model metadata (family,
implementation, class count, input size, weights, label source, predicted index)
is intentionally **not shown** in the normal result — the backend response still
contains it, but the UI omits it so farmers see only the label, confidence,
candidates and probabilities. The alternatives are derived from the
`probabilities` array via a sorted copy (the response is never mutated); entries
with missing/non-finite/negative probabilities are dropped, and no labels,
scores or disease lists are invented. Nothing is spoken automatically — the
section speaker reads the prediction and its uncertainty note only when clicked.

After a successful prediction, the result card shows **Continue exploring this
crop**, which opens **AI Crop Insights** with the diagnosed crop selected and
adds a *diagnosis reference* turn (label + confidence) to the conversation. A
new diagnosis uses the newest `state.lastResult`; earlier reference turns are
snapshots and are not corrupted when you analyze another image.

## 7. AI Crop Insights (continuous session)

`AI Crop Insights` is a single-crop, multi-turn workspace:

- Choose a **crop** for the session; the crop, the turn count and an explicit
  **Session only** badge are shown. State lives in memory for the current tab
  only — it is **not** saved to a server or an account and does not sync across
  devices or reloads.
- The **conversation** lists every turn with a turn number and timestamp:
  questions you asked, the AI answers (or a clear error turn when the feature is
  off or a request fails), diagnosis references, and backend prediction
  summaries. Prior turns stay visible; if a question is superseded, its
  placeholder **Thinking** turn is removed.
- **Charts** stay available throughout the session (see §9).
- **Clear session** empties the conversation.
- Changing the crop is explicit: the earlier turns are cleared and a note says a
  new session started, so answers/data from different crops are never mixed.

### General AI question-answering (`POST /insights/ask`)

The composer connects to the backend's general crop Q&A endpoint
`POST /insights/ask` (see `backend/app/routers/insights.py`). That endpoint is
**disabled by default** and needs a server-side Groq API key, so availability
depends on the operator's configuration:

- Pressing **Ask** sends JSON with the required `crop` and `question`, the
  selected `topic`, and — only when the most recent diagnosis in this tab
  belongs to the **same crop** — a `context` object with the supported fields
  `predicted_label`, `confidence` and `family`. Only the **current** question is
  sent: prior turns stay visible in the tab but are not sent, because the
  contract does not accept conversation history.
- While the request is in flight the Ask button is disabled (label
  **Asking…**), a **Thinking** turn is added, and the status reads *Waiting for
  the backend AI service…*.
- On success the **AI answer** turn shows the returned `answer`, the response
  `disclaimer` and the model id + topic. A response with an empty/missing
  `answer` is shown as an error — nothing is invented.
- Errors surface the standard backend envelope through `describeError`:
  `insights_disabled` and `insights_not_configured` explain that the **operator**
  must enable the feature / set the API key; network, timeout, rate-limit,
  upstream and validation errors show their mapped messages.
- **No API key ever reaches the browser** and no fallback answer is generated.
- A new **Ask**, changing the crop, or **Clear session** aborts any in-flight
  question so a stale answer cannot appear after the conversation changed.
- AI answers are **never spoken automatically** — use a section speaker button
  if you want to hear one.

`/speech/response` is still used only as a *prediction summary* (a backend
transformation of a diagnosis) — it is never presented as general Q&A.

## 8. Text-to-speech: page and section reading (browser)

Two reading modes share the same browser `speechSynthesis` engine, queue and
cancellation safeguards. **All speech is strictly user-initiated** — nothing is
ever spoken by page load, navigation, prediction completion, rendering or the
`MutationObserver`; speech begins only from a click on **Read page** or a
section speaker button.

- **Read page** (top bar) reads **all meaningful visible text in the current
  view** when you click it.
- **Read this section aloud** — a small speaker button on each card, chart,
  result panel and insight turn — reads **only that section's** visible heading,
  explanation and displayed result when you click it.

How they work together:

- Starting either mode **cancels the previous reading** and starts the new one
  cleanly (`generation` token), so no stale `onend`/`onerror` can advance or stop
  a newer reading.
- **Pause/Resume** and **Stop** apply to whichever mode is currently active.
- The status chip shows **Reading page**, **Reading section**, **Paused**,
  **Finished reading**, or an error, and clears when reading stops.
- **Navigating** to another view cancels any active reading (browser and backend)
  and clears the status; it never starts speech.

Reliability measures implemented:

- **Voice loading** — voices are loaded at start-up and on `voiceschanged`; a
  preferred English voice (`en-IN` → `en-GB` → `en-US` → any `en` → first
  available) is selected. If no voice is ready yet, the UI asks you to press the
  button again rather than failing silently.
- **Paused-state reset** — `cancel()` is always followed by clearing the
  engine's paused flag (`resume()` when `paused`), because a paused engine
  queues later `speak()` calls that never play.
- **Sequential chunks** — text is split into ~200-character sentence chunks and
  queued one at a time via `onend`; each chunk carries a generation token.
- **Cancellation** — starting a new reading, navigating, or pressing Stop
  cancels cleanly; repeated clicks are safe.

### Section read controls

- A single shared collector (`collectReadableText`) builds the spoken text. It
  strips buttons, form controls, `hidden`/`aria-hidden` content, scripts, styles
  and `.no-read`, so the read controls themselves, navigation labels, form
  fields, hidden views and hidden result panels are never spoken.
- Controls are real `<button>` elements built with DOM APIs (never `innerHTML`),
  carry the label **Read this section aloud**, and support keyboard activation.
  They are small and unobtrusive (a speaker icon in the section header, or a
  subtle corner control when there is no header) and never make the whole card
  clickable, so selecting a crop, uploading a file, opening a link or submitting
  a form is unaffected.
- A `MutationObserver` re-scans after content changes, so prediction results,
  scheme cards, model cards, insight turns and chart summaries all get a working
  control even when they are re-rendered. The observer **only adds missing
  controls — it never starts speech.** Each section is marked once, so a parent
  section and its children are not read twice by a single activation.

This is **browser** speech, distinct from the backend `/speech/speak` WAV audio
used for the spoken prediction summary when you press **Listen to result**. No
prediction is read automatically. Turning Voice off stops any speech currently
playing. No Hindi or Meitei synthesis is claimed.

Troubleshooting when you hear nothing:

1. Confirm the OS/browser has speech voices installed (some Linux/Chromium
   builds ship none); the status chip reports when none are ready.
2. Press the button again after a moment — voices can load lazily.
3. Unmute the tab/system and ensure the "Voice output" toggle is on.
4. Background/hidden tabs may be throttled by the browser.

## 9. Charts (all real backend data)

Each chart has a title, numeric labels and an explanatory caption, and remains
available while the session is open.

| Chart | Source | Notes |
| --- | --- | --- |
| **Diagnosis probabilities** | `POST /predict` result (this session) | Actual class scores; empty state until you run a diagnosis. |
| **Model class counts** | `GET /models` | Output classes per model; explicitly not a quality measure. |
| **Class-set overlap** | `GET /comparison/{crop}` | Shared / baseline-only / fine-tuned-only label counts. Defaults to the session crop. |

No chart invents prevalence, outbreak, seasonal, treatment-effectiveness or yield
data; requests for such data are answered with an explicit "unavailable"
statement. Missing data shows an empty/unavailable state, never fabricated bars.

## 10. Honest behavior and limitations

- Predictions are model outputs, not confirmed diagnoses; the API `disclaimer` is
  shown with every result. The baseline models report high accuracy on their own
  held-out sets, but those test photos come from the same datasets the model was
  trained on; **real field photos score lower**, so a biologically surprising
  label usually reflects model/domain limits, not a UI bug unless the request or
  response is malformed.
- Scheme results are search leads, not proof of eligibility or that a scheme is
  open. A **verified** badge means the host is an allow-listed government domain
  only.
- Insights state, My Crops, History and Feedback are local to the current tab
  only; nothing is persisted server-side. Voice preference uses `localStorage`.
- Backend speech synthesis supports English and Arabic-Saudi only; the page
  reader and the per-section read controls use the browser voice and claim no
  specific language support.
- No automated browser test suite is bundled; camera capture works only where
  the device/browser supports it, and browser text-to-speech depends on
  voices being installed on the device.
