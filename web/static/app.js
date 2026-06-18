const form = document.querySelector("#predict-form");
const queryInput = document.querySelector("#query");
const statusEl = document.querySelector("#status");
const resultLabel = document.querySelector("#result-label");
const confidenceEl = document.querySelector("#confidence");
const meterFill = document.querySelector("#meter-fill");
const phaseDeltaEl = document.querySelector("#phase-delta");
const copyButton = document.querySelector("#copy-summary");
const methodEl = document.querySelector("#method");
const modelNoteEl = document.querySelector("#model-note");
const contributionsEl = document.querySelector("#contributions");
const featuresEl = document.querySelector("#features");
const rawDataEl = document.querySelector("#raw-data");
const associatedMetaEl = document.querySelector("#associated-meta");
const associatedRowsEl = document.querySelector("#associated-rows");
const competitionTitleEl = document.querySelector("#competition-title");
const competitionCountEl = document.querySelector("#competition-count");
const competitionRowsEl = document.querySelector("#competition-rows");
const riskOverallEl = document.querySelector("#risk-overall");
const riskFactorsEl = document.querySelector("#risk-factors");
const quickStatsEl = document.querySelector("#quick-stats");
const topDriversEl = document.querySelector("#top-drivers");
const confidenceDetailEl = document.querySelector("#confidence-detail");
const comparableRowsEl = document.querySelector("#comparable-rows");
const missingDataEl = document.querySelector("#missing-data");
const reviewFlagsEl = document.querySelector("#review-flags");

const fields = {
  drug: document.querySelector("#trial-drug"),
  nct: document.querySelector("#trial-nct"),
  status: document.querySelector("#trial-status"),
  sponsor: document.querySelector("#trial-sponsor"),
  phase: document.querySelector("#trial-phase"),
  condition: document.querySelector("#trial-condition"),
};

let latestSummary = "";

function fmt(value) {
  if (value === null || value === undefined || value === "") return "NA";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function deltaText(value) {
  const points = Math.round((value || 0) * 100);
  if (points === 0) return "Delta 0pp";
  return `Delta ${points > 0 ? "+" : ""}${points}pp`;
}

function nctLink(nctId) {
  return `<a href="https://clinicaltrials.gov/study/${nctId}" target="_blank" rel="noreferrer">${nctId}</a>`;
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();
  if (!data.trained) {
    statusEl.textContent = "Model not trained yet. Run the training command, then refresh this page.";
    return;
  }
  statusEl.textContent = `Model ready: ${data.task} · Data from ClinicalTrials.gov`;
}

function renderTrial(trial) {
  if (!trial) return;
  fields.drug.textContent = trial.drug || "Unknown drug";
  fields.nct.textContent = trial.nct_id || "Unknown";
  fields.status.textContent = trial.status || "Unknown";
  fields.sponsor.textContent = trial.sponsor || "Unknown";
  fields.phase.textContent = trial.phase || "Unknown";
  fields.condition.textContent = trial.condition || trial.conditions || "Unknown";

  document.querySelectorAll(".timeline li").forEach((item) => {
    item.classList.toggle("active", item.dataset.phase === trial.phase);
  });
}

function renderContributions(contributions) {
  contributionsEl.innerHTML = "";
  (contributions || []).forEach((contribution) => {
    const item = document.createElement("div");
    item.className = "contribution-item";
    item.innerHTML = `<span>${contribution.label}</span><strong>${contribution.value}</strong><small>${contribution.detail}</small>`;
    contributionsEl.appendChild(item);
  });
}

function renderFeatures(features) {
  featuresEl.innerHTML = "";
  (features || []).forEach((feature) => {
    const item = document.createElement("div");
    item.className = "feature-item";
    item.innerHTML = `<strong>${feature.label}</strong><span>${feature.detail}</span>`;
    featuresEl.appendChild(item);
  });
}

function renderRawData(rows) {
  rawDataEl.innerHTML = "";
  (rows || []).forEach((row) => {
    const item = document.createElement("div");
    item.className = "raw-item";
    item.innerHTML = `<span>${row.label}</span><strong>${fmt(row.value)}</strong>`;
    rawDataEl.appendChild(item);
  });
}

function renderAssociated(section) {
  associatedMetaEl.textContent = `Total: ${section.total || 0} · Phases: ${section.phases || "NA"}`;
  associatedRowsEl.innerHTML = "";
  (section.rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fmt(row.phase)}</td>
      <td>${nctLink(row.nct_id)}</td>
      <td>${fmt(row.title)}</td>
      <td>${fmt(row.status)}</td>
      <td>${fmt(row.enrollment)}</td>
      <td>${fmt(row.start)}</td>
    `;
    associatedRowsEl.appendChild(tr);
  });
}

function renderCompetition(section) {
  competitionTitleEl.textContent = section.title || "Competitive Landscape";
  competitionCountEl.textContent = `${section.count || 0} trials`;
  competitionRowsEl.innerHTML = "";
  (section.rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.className = row.is_current ? "current-row" : "";
    tr.innerHTML = `
      <td>${fmt(row.drug)}${row.is_current ? '<span class="you-badge">YOU</span>' : ""}</td>
      <td>${fmt(row.sponsor)}</td>
      <td>${fmt(row.phase)}</td>
      <td>${fmt(row.status)}</td>
      <td>${nctLink(row.nct_id)}</td>
    `;
    competitionRowsEl.appendChild(tr);
  });
}

function renderRisk(section) {
  riskOverallEl.textContent = section.overall || "Risk unavailable";
  riskFactorsEl.innerHTML = "";
  (section.factors || []).forEach((factor) => {
    const item = document.createElement("div");
    item.className = "risk-item";
    item.innerHTML = `<div><strong>${factor.name}</strong><span>${factor.detail}</span></div><em class="${factor.level}">${factor.level}</em>`;
    riskFactorsEl.appendChild(item);
  });

  quickStatsEl.innerHTML = "";
  (section.quick_stats || []).forEach((stat) => {
    const item = document.createElement("div");
    item.className = "stat-item";
    item.innerHTML = `<span>${stat.label}</span><strong>${stat.value}</strong><small>${stat.benchmark}</small><em>${stat.rating}</em>`;
    quickStatsEl.appendChild(item);
  });
}

function renderScoreExplanation(section) {
  topDriversEl.innerHTML = "";
  (section.top_drivers || []).forEach((driver) => {
    const item = document.createElement("div");
    item.className = `driver-item ${driver.direction || "neutral"}`;
    item.innerHTML = `<div><strong>${driver.name}</strong><span>${driver.detail}</span></div><em>${driver.impact}</em>`;
    topDriversEl.appendChild(item);
  });

  const confidence = section.confidence || {};
  confidenceDetailEl.innerHTML = `<strong>${confidence.label || "NA"}</strong><span>${confidence.detail || ""}</span>`;

  comparableRowsEl.innerHTML = "";
  (section.comparable_historical_trials || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${nctLink(row.nct_id)}</td>
      <td>${fmt(row.title)}</td>
      <td>${fmt(row.phase)}</td>
      <td>${fmt(row.enrollment)}</td>
      <td>${fmt(row.outcome)}</td>
    `;
    comparableRowsEl.appendChild(tr);
  });

  missingDataEl.innerHTML = "";
  (section.missing_data || []).forEach((item) => {
    const pill = document.createElement("span");
    pill.textContent = item;
    missingDataEl.appendChild(pill);
  });

  reviewFlagsEl.innerHTML = "";
  (section.review_flags || []).forEach((flag) => {
    const item = document.createElement("div");
    item.className = "flag-item";
    item.textContent = flag;
    reviewFlagsEl.appendChild(item);
  });
}

function buildSummary(data) {
  const trial = data.trial || {};
  const explanation = data.explanation || {};
  return [
    `${trial.drug || data.query}`,
    `${trial.nct_id || ""}`,
    `${trial.status || ""}`,
    `${trial.sponsor || ""}`,
    `${trial.phase || ""}`,
    `${trial.condition || ""}`,
    "",
    `${data.approval_percent || Math.round((data.approval_probability || 0) * 100)}%`,
    `${data.confidence || ""}`,
    "ML Live",
    `Phase avg ${percent(explanation.phase_avg)} · ${deltaText(explanation.delta)}`,
  ].join("\n");
}

function renderPrediction(data) {
  const explanation = data.explanation || {};
  const approvalProbability = data.approval_probability || 0;

  renderTrial(data.trial);
  resultLabel.textContent = `${data.approval_percent || Math.round(approvalProbability * 100)}%`;
  confidenceEl.textContent = data.confidence || "LOW CONF";
  meterFill.style.width = `${Math.max(3, Math.round(approvalProbability * 100))}%`;
  phaseDeltaEl.textContent = `Phase avg ${percent(explanation.phase_avg)} · ${deltaText(explanation.delta)}`;
  statusEl.textContent = `${data.source} for "${data.query}"`;

  renderContributions(explanation.contributions);
  methodEl.textContent = explanation.method || "";
  modelNoteEl.textContent = explanation.model_note || "";
  renderFeatures(explanation.features);
  renderRawData(data.raw_data);
  renderScoreExplanation(data.score_explanation || {});
  renderAssociated(data.associated_trials || {});
  renderCompetition(data.competitive_landscape || {});
  renderRisk(data.risk_assessment || {});
  latestSummary = buildSummary(data);
}

async function runPrediction(query) {
  resultLabel.textContent = "--%";
  confidenceEl.textContent = "Calculating";
  meterFill.style.width = "12%";
  statusEl.textContent = "Fetching ClinicalTrials.gov data and running inference...";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await response.json();

  if (!response.ok) {
    resultLabel.textContent = "--%";
    confidenceEl.textContent = "Unavailable";
    statusEl.textContent = data.error || "Prediction failed.";
    meterFill.style.width = "0%";
    return;
  }
  renderPrediction(data);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query) runPrediction(query);
});

copyButton.addEventListener("click", async () => {
  if (!latestSummary) return;
  await navigator.clipboard.writeText(latestSummary);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy summary";
  }, 1200);
});

refreshStatus()
  .then(() => runPrediction(queryInput.value.trim()))
  .catch(() => {
    statusEl.textContent = "Could not reach the prediction service.";
  });
