# Interactive House Price Estimator — Frontend Design

**Date:** 2026-06-29
**Status:** Approved (pending spec review)

## Goal

Make the project practical for a real person by adding a web UI on top of the
existing Flask ML API. A visitor fills in a handful of intuitive house
characteristics and sees an estimated price, with interactive feedback. No new
services or build tooling — the existing Flask app serves the page.

## Scope

In scope:
- A single interactive page served by Flask.
- A new `GET /metadata` endpoint to drive the UI from real data (no hardcoding).
- Repurposing `GET /` to serve the page.
- Backend tests for the new/changed routes.

Out of scope (YAGNI):
- React/Vite/Node or any frontend build step.
- A JS test framework.
- Authentication, persistence, multi-page navigation.
- Changes to the ML model, `/predict`, or `/health` behavior.

## Architecture

A single page served by the existing Flask app — no build step, no new services.

- **`GET /`** returns `templates/index.html` (replaces the current plain-text
  home message).
- **Static assets:** `static/style.css`, `static/app.js`.
- **`POST /predict`** unchanged — the JS calls it as-is.
- **`GET /health`** unchanged.
- **`GET /metadata`** new — see below.
- Flask gains standard `templates/` and `static/` folders.

### The 4 interactive functions

These are the page behaviors the user requested:

1. **Preset example houses** — 3 buttons (Starter / Family / Luxury) that fill
   the form instantly.
2. **Live estimate** — price updates automatically as inputs change (debounced),
   not only on a button press.
3. **Price visualization** — a horizontal gauge showing where the predicted
   price sits within the dataset's price range.
4. **Input guidance** — sliders and dropdowns with sensible ranges/labels so
   inputs are realistic and hard to break.

### Form fields (8)

| Field          | Model key      | Control                         |
|----------------|----------------|---------------------------------|
| Overall Quality| `OverallQual`  | slider 1–10                     |
| Living Area    | `GrLivArea`    | number input (sq ft)            |
| Basement Area  | `TotalBsmtSF`  | number input (sq ft)            |
| Year Built     | `YearBuilt`    | number input                    |
| Bedrooms       | `BedroomAbvGr` | slider 0–8                      |
| Full Baths     | `FullBath`     | slider 0–4                      |
| Garage Cars    | `GarageCars`   | slider 0–4                      |
| Neighborhood   | `Neighborhood` | dropdown (from `/metadata`)     |

Unspecified/empty fields are omitted from the payload; the backend fills them
with training defaults (existing behavior).

### Page layout (one screen, three zones)

1. **Form** — the 8 guided fields above.
2. **Presets bar** — 3 buttons that fill the form.
3. **Result panel** — predicted price (large number) + horizontal gauge scaled
   to the dataset price range.

## Data flow

- **On load:** JS fetches `GET /metadata` → populates the Neighborhood dropdown,
  slider/number bounds, and the gauge scale.
- **On input change (debounced ~400ms) and on preset click:** JS POSTs the 8
  fields to `POST /predict` → renders price + animates gauge. This is the live
  estimate.

## New backend endpoint: `GET /metadata`

Computed once at startup (from the dataset/defaults) so the frontend isn't
hardcoded. Returns:

```json
{
  "neighborhoods": ["Blmngtn", "Blueste", "..."],
  "price_range": { "min": 34900, "max": 755000 },
  "field_ranges": {
    "GrLivArea":   { "min": 334,  "max": 5642 },
    "TotalBsmtSF": { "min": 0,    "max": 6110 },
    "YearBuilt":   { "min": 1872, "max": 2010 }
  }
}
```

- `neighborhoods`: sorted unique values of `Neighborhood`.
- `price_range`: min/max of `SalePrice`.
- `field_ranges`: min/max for the numeric form fields with free ranges.

Slider fields (OverallQual, Bedrooms, FullBath, GarageCars) use fixed UI ranges
and don't need metadata.

Implementation note: compute from the training data via the existing
`preprocessing` module at import/startup and cache in module state, consistent
with how `predict.py` loads artifacts once.

## Error handling

- **Metadata fetch fails** → form still works with sensible hardcoded fallback
  ranges; gauge shows a neutral scale. Show a small non-blocking notice.
- **`/predict` fails or returns 4xx/5xx** → result panel shows a friendly
  "Couldn't estimate — check inputs" message, not a raw error. Keep the last
  good price visible if possible.
- **Debounce + in-flight guard** → track the latest request and ignore stale
  responses so a slow earlier call can't overwrite a newer result.
- **Empty/invalid field** → JS omits empty fields from the payload; non-numeric
  input is blocked by HTML input types.

## Testing

- **Backend (pytest):**
  - `GET /` returns 200 and HTML.
  - `GET /metadata` returns 200 with keys `neighborhoods`, `price_range`,
    `field_ranges`; `neighborhoods` non-empty; `price_range.min < price_range.max`.
  - Existing `/predict` and `/health` tests remain untouched and passing.
- **Frontend:** no JS test framework (YAGNI). Manual verification checklist:
  - Presets fill the form.
  - Editing an input triggers a debounced live update.
  - Gauge position reflects the predicted price within the range.
  - Error states render gracefully (stop the server / send bad input).

## Files

- `app.py` — repurpose `/`, add `/metadata`.
- `templates/index.html` — new.
- `static/style.css` — new.
- `static/app.js` — new.
- `tests/test_app.py` — add `/` and `/metadata` tests.
- `README.md` — document the web UI and `/metadata`.
