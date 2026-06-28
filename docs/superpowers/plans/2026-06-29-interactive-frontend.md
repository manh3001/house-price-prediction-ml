# Interactive House Price Estimator Frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single interactive web page, served by the existing Flask app, that lets a visitor enter 8 house features and see a live price estimate with presets and a price gauge.

**Architecture:** The existing Flask app serves `templates/index.html` at `GET /` plus `static/` assets. A new `GET /metadata` endpoint feeds real dataset ranges/neighborhoods to the page. Vanilla JS calls the unchanged `POST /predict` with debounced live updates. No build step, no new services.

**Tech Stack:** Flask 3 (render_template, jsonify), pandas/numpy (existing `preprocessing` module), vanilla HTML/CSS/JS, pytest.

## Global Constraints

- Python deps pinned in `requirements.txt`; do not add new runtime dependencies (no Node, no JS framework, no JS test framework).
- Do not change the behavior of `POST /predict` or `GET /health`.
- Flask app object is `app` in `app.py`; prediction logic stays in `predict.py`; shared data logic stays in `preprocessing.py`.
- Tests run with `pytest` from the repo root and import `app`/`preprocessing` as top-level modules (see existing `tests/test_app.py`).
- Dataset is `data/train.csv`; target column is `SalePrice`; categorical neighborhood column is `Neighborhood`.

---

### Task 1: `/metadata` endpoint backed by `preprocessing.compute_metadata`

**Files:**
- Modify: `preprocessing.py` (add `UI_NUMERIC_FIELDS` constant + `compute_metadata`)
- Modify: `app.py` (import metadata, add `/metadata` route)
- Test: `tests/test_preprocessing.py` (unit test for `compute_metadata`)
- Test: `tests/test_app.py` (route test for `/metadata`)

**Interfaces:**
- Consumes: `preprocessing.DATA_PATH`, `preprocessing.TARGET` (existing).
- Produces:
  - `preprocessing.UI_NUMERIC_FIELDS: list[str]` = `["GrLivArea", "TotalBsmtSF", "YearBuilt"]`
  - `preprocessing.compute_metadata(path: str = DATA_PATH) -> dict` returning
    `{"neighborhoods": list[str], "price_range": {"min": int, "max": int},
    "field_ranges": {col: {"min": int, "max": int}}}`
  - `GET /metadata` → JSON of the dict above, computed once at import as `app._METADATA`.

- [ ] **Step 1: Write the failing unit test for `compute_metadata`**

In `tests/test_preprocessing.py`, add:

```python
def test_compute_metadata_structure():
    import preprocessing
    meta = preprocessing.compute_metadata()
    assert set(meta) >= {"neighborhoods", "price_range", "field_ranges"}
    assert len(meta["neighborhoods"]) > 0
    assert meta["neighborhoods"] == sorted(meta["neighborhoods"])
    assert meta["price_range"]["min"] < meta["price_range"]["max"]
    for col in ("GrLivArea", "TotalBsmtSF", "YearBuilt"):
        r = meta["field_ranges"][col]
        assert r["min"] <= r["max"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_preprocessing.py::test_compute_metadata_structure -v`
Expected: FAIL with `AttributeError: module 'preprocessing' has no attribute 'compute_metadata'`

- [ ] **Step 3: Implement `compute_metadata` in `preprocessing.py`**

After the `TARGET = "SalePrice"` line, add the constant:

```python
UI_NUMERIC_FIELDS = ["GrLivArea", "TotalBsmtSF", "YearBuilt"]
```

At the end of `preprocessing.py`, add:

```python
def compute_metadata(path: str = DATA_PATH) -> dict:
    """UI metadata derived from the raw dataset: neighborhoods, price and
    numeric-field ranges. Used to drive the frontend without hardcoding."""
    df = pd.read_csv(path)
    neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())
    price_range = {"min": int(df[TARGET].min()), "max": int(df[TARGET].max())}
    field_ranges = {
        col: {"min": int(df[col].min()), "max": int(df[col].max())}
        for col in UI_NUMERIC_FIELDS
    }
    return {
        "neighborhoods": neighborhoods,
        "price_range": price_range,
        "field_ranges": field_ranges,
    }
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `pytest tests/test_preprocessing.py::test_compute_metadata_structure -v`
Expected: PASS

- [ ] **Step 5: Write the failing route test for `/metadata`**

In `tests/test_app.py`, add:

```python
def test_metadata_returns_expected_keys():
    resp = client().get("/metadata")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) >= {"neighborhoods", "price_range", "field_ranges"}
    assert len(data["neighborhoods"]) > 0
    assert data["price_range"]["min"] < data["price_range"]["max"]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/test_app.py::test_metadata_returns_expected_keys -v`
Expected: FAIL with status 404

- [ ] **Step 7: Add the `/metadata` route in `app.py`**

Change the import line and add module-level metadata + route. Replace:

```python
import predict as predictor

app = Flask(__name__)
_VALID_KEYS = set(predictor._defaults.keys())
```

with:

```python
import predict as predictor
import preprocessing

app = Flask(__name__)
_VALID_KEYS = set(predictor._defaults.keys())
_METADATA = preprocessing.compute_metadata()
```

Then add this route after the `health` route:

```python
@app.route("/metadata")
def metadata():
    return jsonify(_METADATA)
```

- [ ] **Step 8: Run both new tests + full suite to verify green**

Run: `pytest -v`
Expected: all tests PASS (existing `/health`, `/predict` tests still pass)

- [ ] **Step 9: Commit**

```bash
git add preprocessing.py app.py tests/test_preprocessing.py tests/test_app.py
git commit -m "feat: add /metadata endpoint for frontend dataset ranges"
```

---

### Task 2: Serve the page structure at `GET /`

**Files:**
- Modify: `app.py` (import `render_template`, repurpose `/` route)
- Create: `templates/index.html`
- Test: `tests/test_app.py` (route test for `/` returns HTML)

**Interfaces:**
- Consumes: Flask `render_template`; `static/style.css` and `static/app.js` are referenced via `url_for` (created in Task 3 — the page renders before they exist; the route test only checks HTML is returned).
- Produces: `GET /` → rendered `index.html` (200, `text/html`) containing the 8 form controls by `id`/`name`, three `.preset` buttons, and the result/gauge elements consumed by Task 3's JS.

- [ ] **Step 1: Write the failing route test for `/`**

In `tests/test_app.py`, add:

```python
def test_home_returns_html_page():
    resp = client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type
    body = resp.get_data(as_text=True)
    assert 'id="house-form"' in body
    assert 'id="Neighborhood"' in body
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_app.py::test_home_returns_html_page -v`
Expected: FAIL — current `/` returns the plain string `"House Price Prediction API is running!"`, so `'id="house-form"'` is not found.

- [ ] **Step 3: Create `templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>House Price Estimator</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <main class="container">
    <header class="header">
      <h1>House Price Estimator</h1>
      <p class="subtitle">Adjust the features below to see an estimated sale price.</p>
    </header>

    <p id="notice" class="notice" hidden></p>

    <section class="presets" aria-label="Example houses">
      <span class="presets__label">Try an example:</span>
      <button type="button" class="preset" data-preset="starter">Starter home</button>
      <button type="button" class="preset" data-preset="family">Family home</button>
      <button type="button" class="preset" data-preset="luxury">Luxury home</button>
    </section>

    <div class="layout">
      <form id="house-form" class="form">
        <div class="field">
          <label for="OverallQual">Overall quality <output id="OverallQual-out"></output></label>
          <input type="range" id="OverallQual" name="OverallQual" min="1" max="10" step="1" value="6">
        </div>
        <div class="field">
          <label for="GrLivArea">Above-ground living area (sq ft)</label>
          <input type="number" id="GrLivArea" name="GrLivArea" min="334" max="6000" step="10" value="1500">
        </div>
        <div class="field">
          <label for="TotalBsmtSF">Basement area (sq ft)</label>
          <input type="number" id="TotalBsmtSF" name="TotalBsmtSF" min="0" max="6500" step="10" value="900">
        </div>
        <div class="field">
          <label for="YearBuilt">Year built</label>
          <input type="number" id="YearBuilt" name="YearBuilt" min="1872" max="2010" step="1" value="2000">
        </div>
        <div class="field">
          <label for="BedroomAbvGr">Bedrooms <output id="BedroomAbvGr-out"></output></label>
          <input type="range" id="BedroomAbvGr" name="BedroomAbvGr" min="0" max="8" step="1" value="3">
        </div>
        <div class="field">
          <label for="FullBath">Full bathrooms <output id="FullBath-out"></output></label>
          <input type="range" id="FullBath" name="FullBath" min="0" max="4" step="1" value="2">
        </div>
        <div class="field">
          <label for="GarageCars">Garage capacity (cars) <output id="GarageCars-out"></output></label>
          <input type="range" id="GarageCars" name="GarageCars" min="0" max="4" step="1" value="2">
        </div>
        <div class="field">
          <label for="Neighborhood">Neighborhood</label>
          <select id="Neighborhood" name="Neighborhood"></select>
        </div>
      </form>

      <aside class="result" aria-live="polite">
        <p class="result__label">Estimated price</p>
        <p id="price" class="result__price">—</p>
        <div class="gauge">
          <div id="gauge-fill" class="gauge__fill"></div>
          <div id="gauge-marker" class="gauge__marker"></div>
        </div>
        <div class="gauge__scale">
          <span id="gauge-min">—</span>
          <span id="gauge-max">—</span>
        </div>
      </aside>
    </div>
  </main>
  <script src="{{ url_for('static', filename='app.js') }}"></script>
</body>
</html>
```

- [ ] **Step 4: Repurpose the `/` route in `app.py`**

Change the import line:

```python
from flask import Flask, request, jsonify
```

to:

```python
from flask import Flask, request, jsonify, render_template
```

Replace the existing home route:

```python
@app.route("/")
def home():
    return "House Price Prediction API is running!"
```

with:

```python
@app.route("/")
def home():
    return render_template("index.html")
```

- [ ] **Step 5: Run the route test + full suite to verify green**

Run: `pytest -v`
Expected: all PASS, including `test_home_returns_html_page`.

- [ ] **Step 6: Commit**

```bash
git add app.py templates/index.html tests/test_app.py
git commit -m "feat: serve interactive estimator page at GET /"
```

---

### Task 3: Styling and interactive behavior (CSS + JS)

**Files:**
- Create: `static/style.css`
- Create: `static/app.js`

**Interfaces:**
- Consumes: element ids/classes from `templates/index.html` (Task 2): `house-form`, the 8 field ids, `*-out` outputs, `.preset[data-preset]`, `notice`, `price`, `gauge-fill`, `gauge-marker`, `gauge-min`, `gauge-max`; endpoints `GET /metadata` and `POST /predict`.
- Produces: the 4 interactive functions (presets, debounced live estimate, gauge visualization, input guidance). No new HTTP surface. Verified manually.

- [ ] **Step 1: Create `static/style.css`**

```css
:root {
  --bg: #f4f6fb;
  --card: #ffffff;
  --ink: #1f2733;
  --muted: #6b7280;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --border: #e5e7eb;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--ink);
}

.container {
  max-width: 920px;
  margin: 0 auto;
  padding: 2rem 1.25rem 3rem;
}

.header h1 { margin: 0 0 .25rem; font-size: 1.9rem; }
.subtitle { margin: 0 0 1.5rem; color: var(--muted); }

.notice {
  background: #fef3c7;
  border: 1px solid #fde68a;
  color: #92400e;
  padding: .6rem .9rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-size: .9rem;
}

.presets {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: .5rem;
  margin-bottom: 1.5rem;
}
.presets__label { color: var(--muted); font-size: .9rem; margin-right: .25rem; }
.preset {
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--ink);
  padding: .45rem .9rem;
  border-radius: 999px;
  cursor: pointer;
  font-size: .9rem;
  transition: background .15s, border-color .15s;
}
.preset:hover { background: var(--accent-soft); border-color: var(--accent); }

.layout {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 1.5rem;
  align-items: start;
}
@media (max-width: 720px) { .layout { grid-template-columns: 1fr; } }

.form {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem;
  display: grid;
  gap: 1.1rem;
}
.field { display: flex; flex-direction: column; gap: .4rem; }
.field label { font-weight: 600; font-size: .9rem; }
.field output { color: var(--accent); font-weight: 700; }
.field input[type="number"], .field select {
  padding: .5rem .6rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 1rem;
}
.field input[type="range"] { width: 100%; accent-color: var(--accent); }

.result {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem;
  position: sticky;
  top: 1rem;
}
.result__label { margin: 0; color: var(--muted); font-size: .9rem; }
.result__price {
  margin: .25rem 0 1.25rem;
  font-size: 2.4rem;
  font-weight: 800;
  color: var(--accent);
}

.gauge {
  position: relative;
  height: 12px;
  background: var(--accent-soft);
  border-radius: 999px;
  overflow: visible;
}
.gauge__fill {
  height: 100%;
  width: 0;
  background: var(--accent);
  border-radius: 999px;
  transition: width .35s ease;
}
.gauge__marker {
  position: absolute;
  top: 50%;
  left: 0;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #fff;
  border: 3px solid var(--accent);
  transform: translate(-50%, -50%);
  transition: left .35s ease;
}
.gauge__scale {
  display: flex;
  justify-content: space-between;
  margin-top: .5rem;
  color: var(--muted);
  font-size: .8rem;
}
```

- [ ] **Step 2: Create `static/app.js`**

```javascript
const FIELDS = [
  "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt",
  "BedroomAbvGr", "FullBath", "GarageCars", "Neighborhood",
];
const RANGE_FIELDS = ["OverallQual", "BedroomAbvGr", "FullBath", "GarageCars"];

const PRESETS = {
  starter: { OverallQual: 4, GrLivArea: 900, TotalBsmtSF: 600, YearBuilt: 1955,
             BedroomAbvGr: 2, FullBath: 1, GarageCars: 1 },
  family:  { OverallQual: 6, GrLivArea: 1500, TotalBsmtSF: 1000, YearBuilt: 1995,
             BedroomAbvGr: 3, FullBath: 2, GarageCars: 2 },
  luxury:  { OverallQual: 9, GrLivArea: 2800, TotalBsmtSF: 1800, YearBuilt: 2008,
             BedroomAbvGr: 4, FullBath: 3, GarageCars: 3 },
};

const FALLBACK = {
  price_range: { min: 35000, max: 755000 },
  field_ranges: {
    GrLivArea: { min: 334, max: 6000 },
    TotalBsmtSF: { min: 0, max: 6500 },
    YearBuilt: { min: 1872, max: 2010 },
  },
  neighborhoods: ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"],
};

let priceRange = FALLBACK.price_range;
let latestRequestId = 0;
let debounceTimer = null;

const $ = (id) => document.getElementById(id);
const usd = (n) => n.toLocaleString("en-US",
  { style: "currency", currency: "USD", maximumFractionDigits: 0 });

function showNotice(msg) {
  const el = $("notice");
  el.textContent = msg;
  el.hidden = false;
}

function syncOutputs() {
  RANGE_FIELDS.forEach((f) => {
    const out = $(`${f}-out`);
    if (out) out.textContent = $(f).value;
  });
}

function collectPayload() {
  const payload = {};
  FIELDS.forEach((f) => {
    const el = $(f);
    if (!el || el.value === "") return;
    payload[f] = el.value;
  });
  return payload;
}

function renderPrice(price) {
  $("price").textContent = usd(price);
  const { min, max } = priceRange;
  const pct = Math.max(0, Math.min(100, ((price - min) / (max - min)) * 100));
  $("gauge-fill").style.width = `${pct}%`;
  $("gauge-marker").style.left = `${pct}%`;
}

async function estimate() {
  const requestId = ++latestRequestId;
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (requestId !== latestRequestId) return; // ignore stale response
    if (!resp.ok) throw new Error("predict failed");
    const data = await resp.json();
    $("notice").hidden = true;
    renderPrice(data.predicted_price);
  } catch (err) {
    if (requestId !== latestRequestId) return;
    showNotice("Couldn't estimate — check your inputs and try again.");
  }
}

function scheduleEstimate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(estimate, 400);
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;
  Object.entries(preset).forEach(([k, v]) => { if ($(k)) $(k).value = v; });
  syncOutputs();
  estimate();
}

function applyMetadata(meta) {
  priceRange = meta.price_range || FALLBACK.price_range;
  $("gauge-min").textContent = usd(priceRange.min);
  $("gauge-max").textContent = usd(priceRange.max);

  const ranges = meta.field_ranges || {};
  Object.entries(ranges).forEach(([f, r]) => {
    const el = $(f);
    if (el) { el.min = r.min; el.max = r.max; }
  });

  const select = $("Neighborhood");
  select.innerHTML = "";
  (meta.neighborhoods || []).forEach((n) => {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    select.appendChild(opt);
  });
}

async function init() {
  syncOutputs();
  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });
  $("house-form").addEventListener("input", () => {
    syncOutputs();
    scheduleEstimate();
  });

  try {
    const resp = await fetch("/metadata");
    if (!resp.ok) throw new Error("metadata failed");
    applyMetadata(await resp.json());
  } catch (err) {
    applyMetadata(FALLBACK);
    showNotice("Using approximate ranges (couldn't load dataset metadata).");
  }

  estimate();
}

document.addEventListener("DOMContentLoaded", init);
```

- [ ] **Step 3: Manual verification — start the server**

Run: `python app.py`
Open: `http://127.0.0.1:5000/`

- [ ] **Step 4: Manual verification checklist**

Confirm each:
- Page loads; Neighborhood dropdown is populated (not empty).
- An initial price shows on load and the gauge marker sits within the bar.
- Moving the Overall Quality slider updates its `output` number immediately and, after ~400ms, the price + gauge update.
- Clicking each preset (Starter / Family / Luxury) fills the form and updates the price; Luxury shows a clearly higher price than Starter.
- Stop the server, change an input → the friendly notice "Couldn't estimate…" appears (no raw error / no crash). Restart server, change input → notice clears and price updates.

- [ ] **Step 5: Commit**

```bash
git add static/style.css static/app.js
git commit -m "feat: interactive estimator UI (presets, live estimate, gauge)"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing. Documents the web UI and `/metadata` endpoint added in Tasks 1–3.
- Produces: nothing consumed by code.

- [ ] **Step 1: Update the "Deploy Model as API" / usage sections of `README.md`**

Under the "Deploy Model as API" section, after the `Server: http://127.0.0.1:5000` line, add:

```markdown
### Web UI

Open `http://127.0.0.1:5000/` in a browser for an interactive estimator: adjust
8 key house features (or load a preset) and see a live price estimate with a
gauge showing where it sits in the dataset's price range. The page is served by
the same Flask app — no separate frontend build.
```

In the "API Usage" section, after the **Health check** block, add:

```markdown
**Metadata**

`GET /metadata` -> `{ "neighborhoods": [...], "price_range": {min, max}, "field_ranges": {...} }`

Drives the web UI's dropdown, input ranges, and price gauge from the dataset.
```

- [ ] **Step 2: Verify the full suite still passes**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document web UI and /metadata endpoint"
```

---

## Self-Review Notes

- **Spec coverage:** 8 fields (Task 2 HTML) ✓; 4 interactive functions — presets (Task 3 `applyPreset`), live estimate (`scheduleEstimate`/debounce), gauge (`renderPrice`), input guidance (slider/number/select controls + metadata ranges) ✓; `/metadata` endpoint (Task 1) ✓; repurposed `/` (Task 2) ✓; `/predict` and `/health` untouched (Global Constraints) ✓; error handling — metadata fallback, friendly predict error, in-flight guard (Task 3 JS) ✓; testing — `/` and `/metadata` pytest tests (Tasks 1–2), manual frontend checklist (Task 3) ✓; README (Task 4) ✓.
- **Placeholder scan:** none — all code blocks complete.
- **Type consistency:** `compute_metadata` dict shape identical across `preprocessing.py`, the route, both tests, and the JS `applyMetadata`/`FALLBACK`. Element ids match between `index.html` and `app.js`.
