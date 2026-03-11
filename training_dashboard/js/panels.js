const REQUIRED_CHECKLIST_TITLES = [
  "Check trained checkpoints",
  "Check PatchTST scaler artifact",
  "Check TensorBoard logs",
  "Check final training configs",
  "Check backup bundle",
  "Run scoring smoke test for both streams",
  "Check split policy documentation"
];

const READINESS_MAP = [
  { key: "checkpoints_ready", label: "checkpoints" },
  { key: "scaler_ready", label: "scaler" },
  { key: "logs_ready", label: "logs" },
  { key: "backup_ready", label: "backup" }
];

function setSectionHidden(element, hidden) {
  if (!element) {
    return;
  }
  element.classList.toggle("is-hidden", hidden);
}

function createChecklistRow(item) {
  const row = document.createElement("article");
  row.className = "check-item";
  row.dataset.state = item.passed ? "pass" : "fail";

  const state = item.passed ? "PASS" : "FAIL";
  const mark = item.passed ? "[v]" : "[ ]";

  const title = document.createElement("div");
  title.className = "check-item-title";
  title.textContent = `${item.index}. ${mark} ${state} - ${item.title}`;

  row.append(title);

  const tooltipParts = [];
  if (item.detail) {
    tooltipParts.push(`detail: ${item.detail}`);
  }
  if (item.hint) {
    tooltipParts.push(`hint: ${item.hint}`);
  }
  if (tooltipParts.length > 0) {
    row.title = tooltipParts.join("\n");
  }

  return row;
}

function createReadinessCard(label, ready) {
  const card = document.createElement("article");
  card.className = "ready-card";
  card.dataset.state = ready ? "ready" : "missing";

  const name = document.createElement("p");
  name.className = "ready-label";
  name.textContent = label;

  const value = document.createElement("strong");
  value.className = "ready-value";
  value.textContent = ready ? "ready" : "missing";

  card.append(name, value);
  return card;
}

function normalizeChecklist(checklist) {
  if (!Array.isArray(checklist)) {
    return [];
  }

  const sorted = [...checklist].sort((a, b) => Number(a.index) - Number(b.index));
  return sorted.filter((item) => REQUIRED_CHECKLIST_TITLES.includes(item.title));
}

function drawLossChart(canvas, series) {
  const ctx = canvas?.getContext("2d");
  if (!ctx || !canvas) {
    return;
  }

  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(8, 17, 35, 0.95)";
  ctx.fillRect(0, 0, width, height);

  if (!Array.isArray(series) || series.length < 2) {
    ctx.fillStyle = "rgba(163, 183, 220, 0.95)";
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillText("No loss history", 12, 24);
    return;
  }

  const pad = { left: 30, right: 10, top: 12, bottom: 22 };
  const plotW = Math.max(width - pad.left - pad.right, 10);
  const plotH = Math.max(height - pad.top - pad.bottom, 10);

  const epochs = series.map((p) => Number(p.epoch));
  const train = series.map((p) => Number(p.train_loss));
  const val = series.map((p) => Number(p.val_loss));
  const all = train.concat(val).filter((v) => Number.isFinite(v));
  if (all.length === 0) {
    return;
  }

  let minY = Math.min(...all);
  let maxY = Math.max(...all);
  if (Math.abs(maxY - minY) < 1e-9) {
    maxY += 1.0;
    minY -= 1.0;
  }

  const minX = Math.min(...epochs);
  const maxX = Math.max(...epochs);
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1e-6);

  const toX = (x) => pad.left + ((x - minX) / spanX) * plotW;
  const toY = (y) => pad.top + (1 - (y - minY) / spanY) * plotH;

  ctx.strokeStyle = "rgba(90, 134, 210, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH);
  ctx.lineTo(width - pad.right, pad.top + plotH);
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.stroke();

  const drawLine = (values, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((y, idx) => {
      const x = toX(epochs[idx]);
      const py = toY(y);
      if (idx === 0) {
        ctx.moveTo(x, py);
      } else {
        ctx.lineTo(x, py);
      }
    });
    ctx.stroke();
  };

  drawLine(train, "#39c7ff");
  drawLine(val, "#ff9f43");

  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = "rgba(171, 189, 224, 0.95)";
  ctx.fillText("train", pad.left, height - 7);
  ctx.fillStyle = "#39c7ff";
  ctx.fillRect(pad.left + 36, height - 14, 10, 2);
  ctx.fillStyle = "rgba(171, 189, 224, 0.95)";
  ctx.fillText("val", pad.left + 56, height - 7);
  ctx.fillStyle = "#ff9f43";
  ctx.fillRect(pad.left + 80, height - 14, 10, 2);
}

function normalizePath(pathValue) {
  if (typeof pathValue !== "string") {
    return "";
  }
  return pathValue.trim();
}

function toHref(pathValue, repoRoot) {
  const raw = normalizePath(pathValue);
  if (!raw) {
    return "";
  }

  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("file://")) {
    return raw;
  }

  let normalized = raw.replace(/\\/g, "/");
  const root = normalizePath(repoRoot).replace(/\\/g, "/");
  if (root && normalized.startsWith(root)) {
    normalized = normalized.slice(root.length).replace(/^\/+/, "");
  }

  if (!normalized) {
    return "";
  }

  if (normalized.startsWith("training_dashboard/")) {
    return `./${normalized.slice("training_dashboard/".length)}`;
  }
  if (normalized.startsWith("data/")) {
    return `./${normalized}`;
  }
  if (normalized.startsWith("./") || normalized.startsWith("../")) {
    return normalized;
  }
  if (normalized.startsWith("/")) {
    return normalized;
  }
  return `../${normalized}`;
}

function createQuickLinkRow({ label, path, exists }, repoRoot) {
  const row = document.createElement("article");
  row.className = "quick-link-row";
  const knownExists = typeof exists === "boolean";
  row.dataset.state = !knownExists ? "unknown" : exists ? "ready" : "missing";

  const title = document.createElement("p");
  title.className = "quick-link-label";
  title.textContent = label;

  const rawPath = normalizePath(path);
  const href = toHref(rawPath, repoRoot);

  let valueNode;
  if (href) {
    const link = document.createElement("a");
    link.className = "quick-link-anchor";
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = rawPath;
    valueNode = link;
  } else {
    const placeholder = document.createElement("span");
    placeholder.className = "quick-link-anchor is-disabled";
    placeholder.textContent = rawPath || "path unavailable";
    valueNode = placeholder;
  }

  row.append(title, valueNode);
  return row;
}

function buildTrainingQuickLinks(runtimeState, runtimeStatePath, runHistoryPath) {
  const artifacts = runtimeState?.artifacts || {};
  const links = [
    {
      label: "dashboard state",
      path: runtimeStatePath || "training_dashboard/data/dashboard-state.json",
      exists: true,
    },
    {
      label: "run index",
      path: runHistoryPath || "training_dashboard/data/runs/index.json",
      exists: true,
    },
    {
      label: "patchtst checkpoint",
      path: artifacts?.checkpoints?.patchtst?.path,
      exists: artifacts?.checkpoints?.patchtst?.exists,
    },
    {
      label: "swinmae checkpoint",
      path: artifacts?.checkpoints?.swinmae?.path,
      exists: artifacts?.checkpoints?.swinmae?.exists,
    },
    {
      label: "scaler artifact",
      path: artifacts?.scaler?.path,
      exists: artifacts?.scaler?.exists,
    },
    {
      label: "patchtst logs",
      path: artifacts?.logs?.patchtst?.path,
      exists: artifacts?.logs?.patchtst?.exists,
    },
    {
      label: "swinmae logs",
      path: artifacts?.logs?.swinmae?.path,
      exists: artifacts?.logs?.swinmae?.exists,
    },
  ];

  if (Array.isArray(artifacts?.backup?.files) && artifacts.backup.files.length > 0) {
    const firstBundle = artifacts.backup.files[0];
    links.push({
      label: "backup bundle",
      path: firstBundle?.path,
      exists: true,
    });
  }

  const dedupe = new Set();
  return links
    .filter((item) => normalizePath(item.path))
    .filter((item) => {
      const key = `${item.label}::${normalizePath(item.path)}`;
      if (dedupe.has(key)) {
        return false;
      }
      dedupe.add(key);
      return true;
    });
}

function toFiniteNumber(value) {
  const numberValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function formatMetric(value, digits = 3) {
  const numeric = toFiniteNumber(value);
  if (numeric == null) {
    return "n/a";
  }
  return numeric.toFixed(digits);
}

function createDecisionSummaryCard(label, value, state = "neutral") {
  const card = document.createElement("article");
  card.className = "decision-summary-card";
  card.dataset.state = state;

  const name = document.createElement("p");
  name.className = "decision-summary-label";
  name.textContent = label;

  const metric = document.createElement("strong");
  metric.className = "decision-summary-value";
  metric.textContent = value;

  card.append(name, metric);
  return card;
}

function createEventPreviewRow(event) {
  const row = document.createElement("article");
  row.className = "decision-event-item";
  row.dataset.state = event?.decision || "normal";

  const header = document.createElement("div");
  header.className = "decision-event-head";

  const badge = document.createElement("span");
  badge.className = "decision-event-badge";
  badge.textContent = event?.decision || "unknown";

  const score = document.createElement("span");
  score.className = "decision-event-score";
  score.textContent = `score ${formatMetric(event?.fused_score, 4)}`;

  header.append(badge, score);

  const meta = document.createElement("p");
  meta.className = "decision-event-meta";
  const eventId = event?.event_id || "event";
  const timestamp = event?.timestamp || "timestamp unavailable";
  meta.textContent = `${eventId} · ${timestamp}`;

  const reason = document.createElement("p");
  reason.className = "decision-event-reason";
  reason.textContent = event?.reason || "No reason available.";

  row.append(header, meta, reason);
  return row;
}

function drawScoreThresholdChart(canvas, chartPayload) {
  const ctx = canvas?.getContext("2d");
  if (!ctx || !canvas) {
    return;
  }

  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(8, 17, 35, 0.95)";
  ctx.fillRect(0, 0, width, height);

  const fused = Array.isArray(chartPayload?.fused_score)
    ? chartPayload.fused_score.map((value) => toFiniteNumber(value))
    : [];
  const patch = Array.isArray(chartPayload?.stream_scores?.patchtst)
    ? chartPayload.stream_scores.patchtst.map((value) => toFiniteNumber(value))
    : [];
  const swin = Array.isArray(chartPayload?.stream_scores?.swinmae)
    ? chartPayload.stream_scores.swinmae.map((value) => toFiniteNumber(value))
    : [];

  const warnThreshold = toFiniteNumber(chartPayload?.thresholds?.warn);
  const anomalyThreshold = toFiniteNumber(chartPayload?.thresholds?.anomaly);
  const validFused = fused.filter((value) => value != null);
  if (validFused.length === 0) {
    ctx.fillStyle = "rgba(163, 183, 220, 0.95)";
    ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillText("No decision trend", 12, 24);
    return;
  }

  const allSeries = [
    ...validFused,
    ...patch.filter((value) => value != null),
    ...swin.filter((value) => value != null),
  ];
  if (warnThreshold != null) {
    allSeries.push(warnThreshold);
  }
  if (anomalyThreshold != null) {
    allSeries.push(anomalyThreshold);
  }

  const pad = { left: 34, right: 12, top: 12, bottom: 32 };
  const plotW = Math.max(width - pad.left - pad.right, 10);
  const plotH = Math.max(height - pad.top - pad.bottom, 10);
  const maxIndex = Math.max(validFused.length - 1, 1);

  let minY = Math.min(...allSeries);
  let maxY = Math.max(...allSeries);
  if (Math.abs(maxY - minY) < 1e-9) {
    maxY += 1.0;
    minY -= 1.0;
  }
  const paddingY = Math.max((maxY - minY) * 0.12, 0.05);
  minY -= paddingY;
  maxY += paddingY;
  const spanY = Math.max(maxY - minY, 1e-6);

  const toX = (index) => pad.left + (index / maxIndex) * plotW;
  const toY = (value) => pad.top + (1 - (value - minY) / spanY) * plotH;

  ctx.strokeStyle = "rgba(90, 134, 210, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH);
  ctx.lineTo(width - pad.right, pad.top + plotH);
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.stroke();

  const drawThreshold = (value, color, label) => {
    if (value == null) {
      return;
    }
    const y = toY(value);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color;
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.fillText(`${label} ${value.toFixed(3)}`, pad.left + 6, Math.max(y - 6, 14));
  };

  const drawSeries = (values, color, lineWidth) => {
    const points = values
      .map((value, index) =>
        value == null ? null : { index, x: toX(index), y: toY(value), value }
      )
      .filter(Boolean);
    if (points.length === 0) {
      return;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();
    return points;
  };

  drawThreshold(warnThreshold, "#ff9f43", "warn");
  drawThreshold(anomalyThreshold, "#ff5b7a", "anomaly");
  drawSeries(patch, "rgba(57, 199, 255, 0.75)", 1.5);
  drawSeries(swin, "rgba(114, 243, 186, 0.75)", 1.5);
  const fusedPoints = drawSeries(fused, "#ffcd63", 2.4) || [];

  fusedPoints.forEach((point, index) => {
    const decision = Array.isArray(chartPayload?.decision)
      ? chartPayload.decision[point.index]
      : "normal";
    let fill = "#72f3ba";
    if (decision === "warn") {
      fill = "#ff9f43";
    } else if (decision === "anomaly") {
      fill = "#ff5b7a";
    }
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 2.8, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = "rgba(171, 189, 224, 0.95)";
  ctx.fillText("patchtst", pad.left, height - 10);
  ctx.fillStyle = "rgba(57, 199, 255, 0.9)";
  ctx.fillRect(pad.left + 56, height - 16, 12, 2);
  ctx.fillStyle = "rgba(171, 189, 224, 0.95)";
  ctx.fillText("swinmae", pad.left + 80, height - 10);
  ctx.fillStyle = "rgba(114, 243, 186, 0.9)";
  ctx.fillRect(pad.left + 138, height - 16, 12, 2);
  ctx.fillStyle = "rgba(171, 189, 224, 0.95)";
  ctx.fillText("fused", pad.left + 162, height - 10);
  ctx.fillStyle = "#ffcd63";
  ctx.fillRect(pad.left + 198, height - 16, 12, 2);
}

function buildBatchQuickLinks(runtimeState, runtimeStatePath) {
  const repoRoot = runtimeState?.meta?.repo_root || "";
  const artifacts = runtimeState?.artifacts?.reports || {};
  const meta = runtimeState?.meta || {};
  const links = [
    {
      label: "batch state",
      path: meta.dashboard_state_path || runtimeStatePath || "training_dashboard/data/batch-decision-state.json",
      exists: true,
    },
    {
      label: "decision report",
      path: artifacts?.report_json?.path || meta.source_report_path,
      exists: artifacts?.report_json?.exists,
    },
    {
      label: "decision events csv",
      path: artifacts?.events_csv?.path,
      exists: artifacts?.events_csv?.exists,
    },
    {
      label: "chart payload",
      path: artifacts?.chart_json?.path,
      exists: artifacts?.chart_json?.exists,
    },
  ];

  return links
    .filter((item) => normalizePath(item.path))
    .map((item) => ({ ...item, repoRoot }));
}

function renderTrainingPanels({
  runtimeState,
  checklistSummaryEl,
  checklistListEl,
  readinessGridEl,
  quickLinksEl,
  patchCanvasEl,
  swinCanvasEl,
  runtimeStatePath,
  runHistoryPath,
}) {
  const checklist = normalizeChecklist(runtimeState?.checklist || []);
  const passed = checklist.filter((item) => Boolean(item.passed)).length;
  checklistSummaryEl.textContent = `${passed}/${checklist.length || 0}`;

  if (checklist.length === 0) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = "No checklist data loaded.";
    checklistListEl.replaceChildren(empty);
  } else {
    const fragment = document.createDocumentFragment();
    checklist.forEach((item) => fragment.appendChild(createChecklistRow(item)));
    checklistListEl.replaceChildren(fragment);
  }

  const readiness = runtimeState?.artifacts?.readiness || {};
  const readyFragment = document.createDocumentFragment();
  READINESS_MAP.forEach(({ key, label }) => {
    readyFragment.appendChild(createReadinessCard(label, Boolean(readiness[key])));
  });
  readinessGridEl.replaceChildren(readyFragment);

  const repoRoot = runtimeState?.meta?.repo_root || "";
  const quickLinks = buildTrainingQuickLinks(runtimeState, runtimeStatePath, runHistoryPath);
  if (quickLinks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = "No quick links available.";
    quickLinksEl.replaceChildren(empty);
  } else {
    const quickLinksFragment = document.createDocumentFragment();
    quickLinks.forEach((item) => {
      quickLinksFragment.appendChild(createQuickLinkRow(item, repoRoot));
    });
    quickLinksEl.replaceChildren(quickLinksFragment);
  }

  drawLossChart(patchCanvasEl, runtimeState?.metrics?.patchtst?.loss || []);
  drawLossChart(swinCanvasEl, runtimeState?.metrics?.swinmae?.loss || []);
}

function renderBatchDecisionPanels({
  runtimeState,
  batchTotalEventsEl,
  batchSummaryGridEl,
  batchSummaryNoteEl,
  batchEventsPreviewEl,
  batchChartCanvasEl,
  batchLinksEl,
  runtimeStatePath,
}) {
  const summary = runtimeState?.summary || {};
  const counts = summary?.decision_counts || {};
  const totalEvents = Number(summary?.total_events || 0);
  batchTotalEventsEl.textContent = String(totalEvents);

  const summaryCards = document.createDocumentFragment();
  summaryCards.append(
    createDecisionSummaryCard("normal", String(Number(counts?.normal || 0)), "normal"),
    createDecisionSummaryCard("warn", String(Number(counts?.warn || 0)), "warn"),
    createDecisionSummaryCard("anomaly", String(Number(counts?.anomaly || 0)), "anomaly"),
    createDecisionSummaryCard("max fused", formatMetric(summary?.max_fused_score, 4), "neutral"),
    createDecisionSummaryCard("mean fused", formatMetric(summary?.mean_fused_score, 4), "neutral"),
    createDecisionSummaryCard("stream", String(summary?.stream || runtimeState?.meta?.run_id || "-"), "neutral")
  );
  batchSummaryGridEl.replaceChildren(summaryCards);

  if (runtimeState?.meta?.run_id) {
    batchSummaryNoteEl.textContent = `run ${runtimeState.meta.run_id} · events ${totalEvents}`;
  } else {
    batchSummaryNoteEl.textContent = totalEvents > 0 ? "Batch decision summary loaded." : "No batch decision state loaded yet.";
  }

  const previewEvents = Array.isArray(runtimeState?.events_preview) ? runtimeState.events_preview : [];
  if (previewEvents.length === 0) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = "No decision events available.";
    batchEventsPreviewEl.replaceChildren(empty);
  } else {
    const previewFragment = document.createDocumentFragment();
    previewEvents.forEach((item) => previewFragment.appendChild(createEventPreviewRow(item)));
    batchEventsPreviewEl.replaceChildren(previewFragment);
  }

  drawScoreThresholdChart(batchChartCanvasEl, runtimeState?.chart || {});

  const quickLinks = buildBatchQuickLinks(runtimeState, runtimeStatePath);
  if (quickLinks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = "No decision artifacts available.";
    batchLinksEl.replaceChildren(empty);
  } else {
    const linksFragment = document.createDocumentFragment();
    quickLinks.forEach((item) => {
      linksFragment.appendChild(createQuickLinkRow(item, item.repoRoot));
    });
    batchLinksEl.replaceChildren(linksFragment);
  }
}

export function renderDashboardPanels({
  mode,
  runtimeState,
  runtimeStatePath,
  runHistoryPath,
  trainingChecklistCardEl,
  trainingComparisonCardEl,
  trainingReadinessCardEl,
  trainingLossCardEl,
  trainingLinksCardEl,
  batchSummaryCardEl,
  batchChartCardEl,
  batchLinksCardEl,
  checklistSummaryEl,
  checklistListEl,
  readinessGridEl,
  quickLinksEl,
  patchCanvasEl,
  swinCanvasEl,
  batchTotalEventsEl,
  batchSummaryGridEl,
  batchSummaryNoteEl,
  batchEventsPreviewEl,
  batchChartCanvasEl,
  batchLinksEl,
}) {
  const isBatchMode = mode === "batch-decision";

  setSectionHidden(trainingChecklistCardEl, isBatchMode);
  setSectionHidden(trainingComparisonCardEl, isBatchMode);
  setSectionHidden(trainingReadinessCardEl, isBatchMode);
  setSectionHidden(trainingLossCardEl, isBatchMode);
  setSectionHidden(trainingLinksCardEl, isBatchMode);
  setSectionHidden(batchSummaryCardEl, !isBatchMode);
  setSectionHidden(batchChartCardEl, !isBatchMode);
  setSectionHidden(batchLinksCardEl, !isBatchMode);

  if (isBatchMode) {
    renderBatchDecisionPanels({
      runtimeState,
      batchTotalEventsEl,
      batchSummaryGridEl,
      batchSummaryNoteEl,
      batchEventsPreviewEl,
      batchChartCanvasEl,
      batchLinksEl,
      runtimeStatePath,
    });
    return;
  }

  renderTrainingPanels({
    runtimeState,
    checklistSummaryEl,
    checklistListEl,
    readinessGridEl,
    quickLinksEl,
    patchCanvasEl,
    swinCanvasEl,
    runtimeStatePath,
    runHistoryPath,
  });
}
