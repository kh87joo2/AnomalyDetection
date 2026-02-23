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

  // Keep detail/hint as tooltip metadata without rendering extra text lines.
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
  const ctx = canvas.getContext("2d");
  if (!ctx) {
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

function buildQuickLinks(runtimeState) {
  const artifacts = runtimeState?.artifacts || {};
  const links = [
    {
      label: "dashboard state",
      path: "training_dashboard/data/dashboard-state.json",
      exists: true,
    },
    {
      label: "run index",
      path: "training_dashboard/data/runs/index.json",
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

export function renderDashboardPanels({
  runtimeState,
  checklistSummaryEl,
  checklistListEl,
  readinessGridEl,
  quickLinksEl,
  patchCanvasEl,
  swinCanvasEl
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

  if (quickLinksEl) {
    const repoRoot = runtimeState?.meta?.repo_root || "";
    const quickLinks = buildQuickLinks(runtimeState);
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
  }

  drawLossChart(patchCanvasEl, runtimeState?.metrics?.patchtst?.loss || []);
  drawLossChart(swinCanvasEl, runtimeState?.metrics?.swinmae?.loss || []);
}
