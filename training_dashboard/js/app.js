import {
  buildNodeMap,
  renderNodes,
  setNodePosition,
  syncNodeVisibility,
  NODE_DIMENSIONS
} from "./nodes.js";
import {
  setupConnectionLayer,
  setConnectionCanvasSize,
  renderConnections
} from "./connections.js";
import { setupInteractions } from "./drag.js";
import { renderDashboardPanels } from "./panels.js";

const TYPE_META = {
  pi: { label: "PI", color: "#ff5b7a" },
  function: { label: "Function", color: "#4f7cff" },
  agent: { label: "Agent", color: "#ffcd63" },
  tool: { label: "Tool", color: "#72f3ba" }
};

const state = {
  layout: null,
  runtimeLive: null,
  runtime: null,
  runHistoryIndex: [],
  runSnapshots: new Map(),
  activeViewIndex: 0,
  activeTypes: new Set(Object.keys(TYPE_META)),
  nodeMap: new Map(),
  nodeElements: new Map(),
  trainerPollId: null,
  trainerApiAvailable: true,
  trainerLiveNodes: null,
  lastTrainerCompletion: "",
  transform: {
    x: 70,
    y: 50,
    scale: 0.9
  }
};

const titleEl = document.getElementById("dashboard-title");
const subtitleEl = document.getElementById("dashboard-subtitle");
const dateEl = document.getElementById("dashboard-date");
const tabsEl = document.getElementById("view-tabs");
const legendEl = document.getElementById("legend");
const statusEl = document.getElementById("status-text");
const canvasWrapper = document.getElementById("canvas-wrapper");
const graphViewport = document.getElementById("graph-viewport");
const groupLayer = document.getElementById("group-layer");
const connectionLayer = document.getElementById("connection-layer");
const nodeLayer = document.getElementById("node-layer");
const checklistSummaryEl = document.getElementById("checklist-summary");
const checklistListEl = document.getElementById("checklist-list");
const readinessGridEl = document.getElementById("readiness-grid");
const quickLinksEl = document.getElementById("quick-links-list");
const patchLossCanvasEl = document.getElementById("patchtst-loss-canvas");
const swinLossCanvasEl = document.getElementById("swinmae-loss-canvas");
const currentRunSelectEl = document.getElementById("current-run-select");
const baselineRunSelectEl = document.getElementById("baseline-run-select");
const comparisonMetricsEl = document.getElementById("comparison-metrics");
const comparisonNoteEl = document.getElementById("comparison-note");
const trainJobStateEl = document.getElementById("train-job-state");
const patchFileInputEl = document.getElementById("patch-file-input");
const patchUploadBtnEl = document.getElementById("patch-upload-btn");
const patchUploadNoteEl = document.getElementById("patch-upload-note");
const swinFileInputEl = document.getElementById("swin-file-input");
const swinUploadBtnEl = document.getElementById("swin-upload-btn");
const swinUploadNoteEl = document.getElementById("swin-upload-note");
const trainRunBtnEl = document.getElementById("train-run-btn");
const trainRefreshBtnEl = document.getElementById("train-refresh-btn");
const trainStopBtnEl = document.getElementById("train-stop-btn");
const trainJobNoteEl = document.getElementById("train-job-note");
const trainJobLogEl = document.getElementById("train-job-log");

const { lineGroup } = setupConnectionLayer(connectionLayer);
const LIVE_RUN_KEY = "__live__";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getCurrentView() {
  return state.layout.views[state.activeViewIndex];
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setTrainNote(text) {
  if (!trainJobNoteEl) {
    return;
  }
  trainJobNoteEl.textContent = text;
}

function setTrainControlsEnabled(enabled) {
  [
    patchFileInputEl,
    patchUploadBtnEl,
    swinFileInputEl,
    swinUploadBtnEl,
    trainRunBtnEl,
    trainStopBtnEl
  ].forEach((element) => {
    if (element) {
      element.disabled = !enabled;
    }
  });
  if (trainRefreshBtnEl) {
    trainRefreshBtnEl.disabled = false;
  }
}

function setTrainStateBadge(stateName) {
  if (!trainJobStateEl) {
    return;
  }
  const normalized = typeof stateName === "string" ? stateName : "idle";
  trainJobStateEl.dataset.state = normalized;
  trainJobStateEl.textContent = normalized.toUpperCase();
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();

  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (error) {
      payload = { message: text };
    }
  }

  if (!response.ok) {
    const message =
      payload?.error || payload?.message || `Request failed (${response.status} ${response.statusText})`;
    throw new Error(message);
  }

  return payload;
}

function getEffectiveRuntimeNodes() {
  const baseNodes = state.runtime?.nodes && typeof state.runtime.nodes === "object"
    ? state.runtime.nodes
    : {};
  const liveNodes = state.trainerLiveNodes;
  if (!liveNodes || typeof liveNodes !== "object") {
    return baseNodes;
  }
  return {
    ...baseNodes,
    ...liveNodes
  };
}

function refreshNodeStatusView() {
  if (!state.layout || !Array.isArray(state.layout.views) || state.layout.views.length === 0) {
    return;
  }
  const view = getCurrentView();
  applyNodeStatuses(view);
  renderConnections({
    lineGroup,
    connections: view.connections,
    nodeMap: state.nodeMap,
    activeTypes: state.activeTypes,
    runtimeNodes: getEffectiveRuntimeNodes()
  });
}

function applyTransform() {
  graphViewport.style.transform = `translate(${state.transform.x}px, ${state.transform.y}px) scale(${state.transform.scale})`;
}

function formatDate(dateString) {
  const parsed = new Date(dateString);
  if (Number.isNaN(parsed.getTime())) {
    return dateString;
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit"
  }).format(parsed);
}

function renderGroups(groups) {
  const fragment = document.createDocumentFragment();

  for (const group of groups) {
    const section = document.createElement("section");
    section.className = "group-box";
    section.style.left = `${group.x}px`;
    section.style.top = `${group.y}px`;
    section.style.width = `${group.width}px`;
    section.style.height = `${group.height}px`;

    const title = document.createElement("span");
    title.className = "group-title";
    title.textContent = group.label;

    section.appendChild(title);
    fragment.appendChild(section);
  }

  groupLayer.replaceChildren(fragment);
}

function renderTabs() {
  const fragment = document.createDocumentFragment();

  state.layout.views.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab-button";
    button.textContent = item.name;
    button.dataset.viewId = item.id;
    button.setAttribute("aria-selected", index === state.activeViewIndex ? "true" : "false");

    button.addEventListener("click", () => {
      state.activeViewIndex = index;
      state.transform = { x: 70, y: 50, scale: 0.9 };
      renderTabs();
      renderView();
      applyTransform();
      setStatus(`Switched to ${item.name}`);
    });

    fragment.appendChild(button);
  });

  tabsEl.replaceChildren(fragment);
}

function renderLegend() {
  const fragment = document.createDocumentFragment();

  Object.entries(TYPE_META).forEach(([type, meta]) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "legend-chip";
    chip.textContent = meta.label;
    chip.style.borderColor = `${meta.color}80`;
    chip.style.boxShadow = `inset 0 0 0 1px ${meta.color}30`;
    chip.setAttribute("aria-pressed", state.activeTypes.has(type) ? "true" : "false");

    chip.addEventListener("click", () => {
      if (state.activeTypes.has(type)) {
        state.activeTypes.delete(type);
      } else {
        state.activeTypes.add(type);
      }

      chip.setAttribute("aria-pressed", state.activeTypes.has(type) ? "true" : "false");
      syncNodeVisibility({
        nodes: getCurrentView().nodes,
        nodeElements: state.nodeElements,
        activeTypes: state.activeTypes
      });

      renderConnections({
        lineGroup,
        connections: getCurrentView().connections,
        nodeMap: state.nodeMap,
        activeTypes: state.activeTypes,
        runtimeNodes: getEffectiveRuntimeNodes()
      });
      setStatus(`Filter updated: ${meta.label}`);
    });

    fragment.appendChild(chip);
  });

  legendEl.replaceChildren(fragment);
}

function renderView() {
  const view = getCurrentView();

  graphViewport.style.width = `${view.canvas.width}px`;
  graphViewport.style.height = `${view.canvas.height}px`;
  groupLayer.style.width = `${view.canvas.width}px`;
  groupLayer.style.height = `${view.canvas.height}px`;
  nodeLayer.style.width = `${view.canvas.width}px`;
  nodeLayer.style.height = `${view.canvas.height}px`;

  setConnectionCanvasSize({
    svgElement: connectionLayer,
    width: view.canvas.width,
    height: view.canvas.height
  });

  renderGroups(view.groups);
  state.nodeMap = buildNodeMap(view.nodes);
  state.nodeElements = renderNodes({
    nodes: view.nodes,
    nodeLayer,
    activeTypes: state.activeTypes
  });

  renderConnections({
    lineGroup,
    connections: view.connections,
    nodeMap: state.nodeMap,
    activeTypes: state.activeTypes,
    runtimeNodes: getEffectiveRuntimeNodes()
  });

  applyNodeStatuses(view);
}

function applyNodeStatuses(view) {
  const nodeStates = getEffectiveRuntimeNodes();
  view.nodes.forEach((node) => {
    const element = state.nodeElements.get(node.id);
    if (!element) {
      return;
    }
    const runtimeNode = nodeStates[node.id] || null;
    const status = runtimeNode?.status || "idle";
    element.dataset.status = status;
    if (runtimeNode?.message) {
      element.title = runtimeNode.message;
    } else {
      element.removeAttribute("title");
    }
  });
}

function setupGraphInteractions() {
  setupInteractions({
    canvasWrapper,
    getTransform: () => state.transform,
    setTransform: (next) => {
      state.transform = next;
    },
    getNodeById: (nodeId) => state.nodeMap.get(nodeId),
    onNodeDrag: (nodeId, x, y) => {
      const view = getCurrentView();
      const node = state.nodeMap.get(nodeId);

      if (!node) {
        return;
      }

      const maxX = view.canvas.width - NODE_DIMENSIONS.width;
      const maxY = view.canvas.height - NODE_DIMENSIONS.height;

      node.x = Math.round(clamp(x, 0, maxX));
      node.y = Math.round(clamp(y, 0, maxY));

      const nodeElement = state.nodeElements.get(nodeId);
      if (nodeElement) {
        setNodePosition(nodeElement, node.x, node.y);
      }

      renderConnections({
        lineGroup,
        connections: view.connections,
        nodeMap: state.nodeMap,
        activeTypes: state.activeTypes,
        runtimeNodes: getEffectiveRuntimeNodes()
      });
    },
    onTransform: applyTransform,
    setStatus
  });
}

async function loadLayoutData() {
  const response = await fetch("./data/dashboard-layout.json");
  if (!response.ok) {
    throw new Error(`Unable to load dashboard layout: ${response.status}`);
  }

  return response.json();
}

async function loadRuntimeData() {
  const response = await fetch("./data/dashboard-state.json", {
    cache: "no-store"
  });
  if (!response.ok) {
    return null;
  }
  return response.json();
}

async function loadRunHistoryIndex() {
  const response = await fetch("./data/runs/index.json", {
    cache: "no-store"
  });
  if (!response.ok) {
    return [];
  }

  const payload = await response.json();
  if (!payload || !Array.isArray(payload.runs)) {
    return [];
  }

  return payload.runs.filter((item) => {
    if (!item || typeof item !== "object") {
      return false;
    }
    return typeof item.run_id === "string" && typeof item.file === "string";
  });
}

function renderRuntimePanels(runtimeState) {
  renderDashboardPanels({
    runtimeState,
    checklistSummaryEl,
    checklistListEl,
    readinessGridEl,
    quickLinksEl,
    patchCanvasEl: patchLossCanvasEl,
    swinCanvasEl: swinLossCanvasEl
  });
}

function summarizeUploadStream(streamPayload, fallbackText) {
  if (!streamPayload || typeof streamPayload !== "object") {
    return fallbackText;
  }
  const count = Number(streamPayload.count || 0);
  if (count <= 0) {
    return fallbackText;
  }

  const files = Array.isArray(streamPayload.files) ? streamPayload.files : [];
  const names = files.slice(0, 2).map((item) => item?.name).filter(Boolean);
  if (names.length === 0) {
    return `${count} file(s) imported`;
  }
  const suffix = count > names.length ? ` +${count - names.length} more` : "";
  return `${count} file(s): ${names.join(", ")}${suffix}`;
}

function setTrainerUnavailable(message) {
  state.trainerApiAvailable = false;
  state.trainerLiveNodes = null;
  stopTrainerPolling();
  setTrainControlsEnabled(false);
  setTrainStateBadge("offline");
  setTrainNote(message);
  if (trainJobLogEl) {
    trainJobLogEl.textContent = "Training API unavailable.";
  }
  refreshNodeStatusView();
}

async function refreshRuntimeFromFiles() {
  state.runtimeLive = await loadRuntimeData();
  state.runtime = state.runtimeLive;
  state.trainerLiveNodes = null;
  state.runSnapshots.clear();
  await initRunComparisonPanel();
  refreshNodeStatusView();
  renderRuntimePanels(state.runtime);
}

function stopTrainerPolling() {
  if (state.trainerPollId !== null) {
    window.clearInterval(state.trainerPollId);
    state.trainerPollId = null;
  }
}

function startTrainerPolling() {
  if (state.trainerPollId !== null) {
    return;
  }
  state.trainerPollId = window.setInterval(() => {
    refreshTrainerStatus().catch((error) => {
      console.error(error);
    });
  }, 3000);
}

function renderTrainerStatus(payload) {
  const job = payload?.job || {};
  const uploads = payload?.uploads || {};
  const liveNodes =
    job.live_nodes && typeof job.live_nodes === "object" ? job.live_nodes : null;

  state.trainerApiAvailable = true;
  setTrainControlsEnabled(true);
  setTrainStateBadge(job.state || "idle");

  if (patchUploadNoteEl) {
    patchUploadNoteEl.textContent = summarizeUploadStream(
      uploads.patchtst,
      "No PatchTST files imported."
    );
  }
  if (swinUploadNoteEl) {
    swinUploadNoteEl.textContent = summarizeUploadStream(
      uploads.swinmae,
      "No SwinMAE files imported."
    );
  }

  if (trainJobNoteEl) {
    const runId = job.run_id ? ` run_id=${job.run_id}` : "";
    const stepTag =
      job.active_step && job.step_index && job.step_total
        ? ` | step ${job.step_index}/${job.step_total}: ${job.active_step}`
        : "";
    trainJobNoteEl.textContent = `${job.message || "Idle"}${runId}${stepTag}`;
  }

  if (trainJobLogEl) {
    const lines = Array.isArray(job.log_tail) ? job.log_tail : [];
    trainJobLogEl.textContent = lines.length > 0 ? lines.join("\n") : "No logs yet.";
  }

  if (job.state === "running" || job.state === "stopping" || job.state === "failed") {
    state.trainerLiveNodes = liveNodes;
  } else if (job.state === "success") {
    state.trainerLiveNodes = liveNodes;
  } else {
    state.trainerLiveNodes = null;
  }
  refreshNodeStatusView();

  if (job.state === "running" || job.state === "stopping") {
    startTrainerPolling();
  } else {
    stopTrainerPolling();
  }

  const completionToken = `${job.state || ""}:${job.finished_at || ""}:${job.run_id || ""}`;
  if (
    (job.state === "success" || job.state === "failed") &&
    completionToken &&
    completionToken !== state.lastTrainerCompletion
  ) {
    state.lastTrainerCompletion = completionToken;
    if (job.state === "success") {
      refreshRuntimeFromFiles()
        .then(() => {
          setStatus("Training completed and dashboard data refreshed.");
        })
        .catch((error) => {
          console.error(error);
          setStatus("Training completed but dashboard refresh failed.");
        });
    } else {
      setStatus("Training failed. Check runner log panel.");
    }
  }
}

async function refreshTrainerStatus() {
  if (!trainJobStateEl) {
    return;
  }
  try {
    const payload = await requestJson("/api/status", { cache: "no-store" });
    renderTrainerStatus(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setTrainerUnavailable(
      `Training API unavailable. Start server: python3 -m training_dashboard.server (${message})`
    );
  }
}

async function uploadStreamData(stream, inputEl) {
  const files = Array.from(inputEl?.files || []);
  if (files.length === 0) {
    setTrainNote(`Select ${stream} files before import.`);
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const payload = await requestJson(`/api/upload?stream=${stream}&replace=1`, {
    method: "POST",
    body: formData
  });
  if (inputEl) {
    inputEl.value = "";
  }
  renderTrainerStatus(payload);
  setStatus(`${stream} files imported (${files.length})`);
}

async function startTrainingFromDashboard() {
  const payload = await requestJson("/api/train", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      persist_run_history: true,
      validate_skip_smoke: true
    })
  });
  renderTrainerStatus(payload);
  setStatus("Training started from dashboard.");
}

async function stopTrainingFromDashboard() {
  const payload = await requestJson("/api/stop", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: "{}"
  });
  renderTrainerStatus(payload);
  setStatus("Stop requested.");
}

async function initTrainingControls() {
  if (!trainJobStateEl) {
    return;
  }

  if (patchUploadBtnEl) {
    patchUploadBtnEl.addEventListener("click", () => {
      uploadStreamData("patchtst", patchFileInputEl).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setTrainNote(message);
        setStatus("PatchTST import failed.");
      });
    });
  }

  if (swinUploadBtnEl) {
    swinUploadBtnEl.addEventListener("click", () => {
      uploadStreamData("swinmae", swinFileInputEl).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setTrainNote(message);
        setStatus("SwinMAE import failed.");
      });
    });
  }

  if (trainRunBtnEl) {
    trainRunBtnEl.addEventListener("click", () => {
      startTrainingFromDashboard().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setTrainNote(message);
        setStatus("Training start failed.");
      });
    });
  }

  if (trainStopBtnEl) {
    trainStopBtnEl.addEventListener("click", () => {
      stopTrainingFromDashboard().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setTrainNote(message);
        setStatus("Stop request failed.");
      });
    });
  }

  if (trainRefreshBtnEl) {
    trainRefreshBtnEl.addEventListener("click", () => {
      refreshTrainerStatus().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setTrainNote(message);
        setStatus("Status refresh failed.");
      });
    });
  }

  await refreshTrainerStatus();
}

function checklistPassedCount(runtimeState) {
  const checklist = runtimeState?.checklist;
  if (!Array.isArray(checklist)) {
    return 0;
  }
  return checklist.filter((item) => item && item.passed).length;
}

function checklistTotalCount(runtimeState) {
  const checklist = runtimeState?.checklist;
  return Array.isArray(checklist) ? checklist.length : 0;
}

function finalValLoss(runtimeState, stream) {
  const series = runtimeState?.metrics?.[stream]?.loss;
  if (!Array.isArray(series) || series.length === 0) {
    return null;
  }

  for (let idx = series.length - 1; idx >= 0; idx -= 1) {
    const item = series[idx];
    const value = item?.val_loss;
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function formatDelta(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "n/a";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function createComparisonItem(label, value, stateName) {
  const item = document.createElement("article");
  item.className = "comparison-item";

  const key = document.createElement("p");
  key.className = "comparison-key";
  key.textContent = label;

  const val = document.createElement("p");
  val.className = "comparison-value";
  val.dataset.state = stateName;
  val.textContent = value;

  item.append(key, val);
  return item;
}

function setComparisonRows(currentRuntime, baselineRuntime) {
  const fragment = document.createDocumentFragment();

  if (!currentRuntime || !baselineRuntime) {
    fragment.append(
      createComparisonItem("checklist pass delta", "n/a", "neutral"),
      createComparisonItem("patchtst final val loss delta", "n/a", "neutral"),
      createComparisonItem("swinmae final val loss delta", "n/a", "neutral")
    );
    comparisonMetricsEl.replaceChildren(fragment);
    comparisonNoteEl.textContent = "Select current and baseline run snapshots to compare.";
    return;
  }

  const currentPassed = checklistPassedCount(currentRuntime);
  const baselinePassed = checklistPassedCount(baselineRuntime);
  const passDelta = currentPassed - baselinePassed;

  const patchCurrent = finalValLoss(currentRuntime, "patchtst");
  const patchBaseline = finalValLoss(baselineRuntime, "patchtst");
  const patchDelta =
    patchCurrent == null || patchBaseline == null ? null : patchCurrent - patchBaseline;

  const swinCurrent = finalValLoss(currentRuntime, "swinmae");
  const swinBaseline = finalValLoss(baselineRuntime, "swinmae");
  const swinDelta = swinCurrent == null || swinBaseline == null ? null : swinCurrent - swinBaseline;

  fragment.append(
    createComparisonItem(
      "checklist pass delta",
      `${passDelta >= 0 ? "+" : ""}${passDelta} (${currentPassed}/${checklistTotalCount(
        currentRuntime
      )} vs ${baselinePassed}/${checklistTotalCount(baselineRuntime)})`,
      passDelta === 0 ? "neutral" : passDelta > 0 ? "improved" : "regressed"
    ),
    createComparisonItem(
      "patchtst final val loss delta",
      formatDelta(patchDelta),
      patchDelta == null ? "neutral" : patchDelta < 0 ? "improved" : patchDelta > 0 ? "regressed" : "neutral"
    ),
    createComparisonItem(
      "swinmae final val loss delta",
      formatDelta(swinDelta),
      swinDelta == null ? "neutral" : swinDelta < 0 ? "improved" : swinDelta > 0 ? "regressed" : "neutral"
    )
  );

  comparisonMetricsEl.replaceChildren(fragment);
  comparisonNoteEl.textContent =
    "delta = current - baseline (negative loss delta is better, positive pass delta is better).";
}

function selectedOptionText(selectEl) {
  if (!selectEl || selectEl.selectedIndex < 0) {
    return "-";
  }
  return selectEl.options[selectEl.selectedIndex]?.textContent || "-";
}

function fillSelectOptions(selectEl, options, selectedValue) {
  const fragment = document.createDocumentFragment();
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    fragment.appendChild(option);
  });
  selectEl.replaceChildren(fragment);
  selectEl.value = selectedValue;
}

async function loadRunSnapshot(entry) {
  const cacheKey = entry.file;
  if (state.runSnapshots.has(cacheKey)) {
    return state.runSnapshots.get(cacheKey);
  }

  const response = await fetch(`./data/runs/${entry.file}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Unable to load run snapshot: ${entry.file}`);
  }
  const payload = await response.json();
  state.runSnapshots.set(cacheKey, payload);
  return payload;
}

async function resolveRuntimeBySelection(selectionValue) {
  if (selectionValue === LIVE_RUN_KEY) {
    return state.runtimeLive;
  }

  const entry = state.runHistoryIndex.find((item) => item.run_id === selectionValue);
  if (!entry) {
    return null;
  }
  return loadRunSnapshot(entry);
}

async function applyRunSelection() {
  const currentRuntime = await resolveRuntimeBySelection(currentRunSelectEl.value);
  const baselineRuntime = await resolveRuntimeBySelection(baselineRunSelectEl.value);

  state.runtime = currentRuntime || state.runtimeLive;
  renderRuntimePanels(state.runtime);
  refreshNodeStatusView();

  const runtimeDate = state.runtime?.meta?.timestamp || state.layout.meta?.date || "";
  dateEl.textContent = formatDate(runtimeDate);
  setComparisonRows(currentRuntime, baselineRuntime);

  const currentLabel = selectedOptionText(currentRunSelectEl);
  const baselineLabel = selectedOptionText(baselineRunSelectEl);
  setStatus(`Comparison ready. current=${currentLabel}, baseline=${baselineLabel}`);
}

async function initRunComparisonPanel() {
  state.runHistoryIndex = await loadRunHistoryIndex();

  if (state.runHistoryIndex.length === 0) {
    currentRunSelectEl.disabled = true;
    baselineRunSelectEl.disabled = true;
    fillSelectOptions(
      currentRunSelectEl,
      [{ value: LIVE_RUN_KEY, label: "live (dashboard-state.json)" }],
      LIVE_RUN_KEY
    );
    fillSelectOptions(baselineRunSelectEl, [{ value: "", label: "none" }], "");
    setComparisonRows(state.runtimeLive, null);
    return;
  }

  const currentOptions = [
    ...state.runHistoryIndex.map((item) => ({
      value: item.run_id,
      label: `${item.run_id} (${item.timestamp || "no-time"})`,
    })),
  ];
  if (state.runtimeLive) {
    currentOptions.unshift({ value: LIVE_RUN_KEY, label: "live (dashboard-state.json)" });
  }
  const baselineOptions = state.runHistoryIndex.map((item) => ({
    value: item.run_id,
    label: `${item.run_id} (${item.timestamp || "no-time"})`,
  }));

  const preferredCurrent = state.runtimeLive ? LIVE_RUN_KEY : baselineOptions[0]?.value || "";
  const preferredBaseline =
    baselineOptions.find((item) => item.value !== state.runtimeLive?.meta?.run_id)?.value ||
    baselineOptions[0]?.value ||
    "";

  fillSelectOptions(currentRunSelectEl, currentOptions, preferredCurrent);
  fillSelectOptions(baselineRunSelectEl, baselineOptions, preferredBaseline);
  currentRunSelectEl.disabled = false;
  baselineRunSelectEl.disabled = baselineOptions.length === 0;

  currentRunSelectEl.onchange = () => {
    applyRunSelection().catch((error) => {
      setStatus("Failed to load selected current run.");
      console.error(error);
    });
  };
  baselineRunSelectEl.onchange = () => {
    applyRunSelection().catch((error) => {
      setStatus("Failed to load selected baseline run.");
      console.error(error);
    });
  };

  await applyRunSelection();
}

async function init() {
  try {
    const [layout, runtime] = await Promise.all([loadLayoutData(), loadRuntimeData()]);
    state.layout = layout;
    state.runtimeLive = runtime;
    state.runtime = runtime;

    if (!Array.isArray(state.layout.views) || state.layout.views.length === 0) {
      throw new Error("dashboard-layout.json must include at least one view.");
    }

    titleEl.textContent = state.layout.meta?.title || "Training Pipeline Dashboard";
    subtitleEl.textContent = state.layout.meta?.subtitle || "Training flow";

    const runtimeDate = state.runtime?.meta?.timestamp || "";
    const layoutDate = state.layout.meta?.date || "";
    dateEl.textContent = formatDate(runtimeDate || layoutDate);

    renderTabs();
    renderLegend();
    renderView();
    renderRuntimePanels(state.runtime);
    applyTransform();
    setupGraphInteractions();
    await initRunComparisonPanel();
    await initTrainingControls();
    const checklistCount = state.runtime?.checklist?.length || 0;
    if (checklistCount > 0) {
      const passed = state.runtime.checklist.filter((item) => item.passed).length;
      setStatus(`Dashboard ready. checklist ${passed}/${checklistCount}`);
    } else {
      setStatus("Dashboard ready. runtime state not found; graph-only mode.");
    }
  } catch (error) {
    setStatus("Failed to initialize dashboard.");
    console.error(error);
  }
}

init();
