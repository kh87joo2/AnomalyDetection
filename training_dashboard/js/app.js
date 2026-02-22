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

const TYPE_META = {
  pi: { label: "PI", color: "#ff5b7a" },
  function: { label: "Function", color: "#4f7cff" },
  agent: { label: "Agent", color: "#ffcd63" },
  tool: { label: "Tool", color: "#72f3ba" }
};

const state = {
  dashboard: null,
  activeViewIndex: 0,
  activeTypes: new Set(Object.keys(TYPE_META)),
  nodeMap: new Map(),
  nodeElements: new Map(),
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

const { lineGroup } = setupConnectionLayer(connectionLayer);

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getCurrentView() {
  return state.dashboard.views[state.activeViewIndex];
}

function setStatus(text) {
  statusEl.textContent = text;
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

  state.dashboard.views.forEach((item, index) => {
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
        activeTypes: state.activeTypes
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
    activeTypes: state.activeTypes
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
        activeTypes: state.activeTypes
      });
    },
    onTransform: applyTransform,
    setStatus
  });
}

async function loadDashboardData() {
  const response = await fetch("./data/dashboard-layout.json");
  if (!response.ok) {
    throw new Error(`Unable to load dashboard layout: ${response.status}`);
  }

  return response.json();
}

async function init() {
  try {
    state.dashboard = await loadDashboardData();
    if (!Array.isArray(state.dashboard.views) || state.dashboard.views.length === 0) {
      throw new Error("dashboard-layout.json must include at least one view.");
    }

    titleEl.textContent = state.dashboard.meta?.title || "Training Pipeline Dashboard";
    subtitleEl.textContent = state.dashboard.meta?.subtitle || "Training flow";
    dateEl.textContent = formatDate(state.dashboard.meta?.date || "");

    renderTabs();
    renderLegend();
    renderView();
    applyTransform();
    setupGraphInteractions();
    setStatus("Dashboard ready.");
  } catch (error) {
    setStatus("Failed to initialize dashboard.");
    console.error(error);
  }
}

init();
