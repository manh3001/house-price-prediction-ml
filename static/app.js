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
