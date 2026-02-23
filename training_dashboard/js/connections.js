import { NODE_DIMENSIONS } from "./nodes.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function createArrowDefs() {
  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");

  marker.setAttribute("id", "flow-arrow");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("orient", "auto-start-reverse");

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  path.setAttribute("fill", "rgba(136, 220, 255, 0.95)");

  marker.appendChild(path);
  defs.appendChild(marker);

  return defs;
}

export function setupConnectionLayer(svgElement) {
  svgElement.replaceChildren();

  const defs = createArrowDefs();
  const lineGroup = document.createElementNS(SVG_NS, "g");

  svgElement.append(defs, lineGroup);
  return { lineGroup };
}

export function setConnectionCanvasSize({ svgElement, width, height }) {
  svgElement.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svgElement.setAttribute("width", String(width));
  svgElement.setAttribute("height", String(height));
  svgElement.style.width = `${width}px`;
  svgElement.style.height = `${height}px`;
}

function getAnchorPair(source, target) {
  const sourceCenterY = source.y + NODE_DIMENSIONS.height / 2;
  const targetCenterY = target.y + NODE_DIMENSIONS.height / 2;
  const isLeftToRight = source.x <= target.x;

  const startX = isLeftToRight ? source.x + NODE_DIMENSIONS.width : source.x;
  const endX = isLeftToRight ? target.x : target.x + NODE_DIMENSIONS.width;

  return {
    startX,
    startY: sourceCenterY,
    endX,
    endY: targetCenterY
  };
}

function getBezierPath(source, target) {
  const anchors = getAnchorPair(source, target);
  const horizontal = Math.abs(anchors.endX - anchors.startX);
  const vertical = Math.abs(anchors.endY - anchors.startY);
  const direction = anchors.startX <= anchors.endX ? 1 : -1;
  const curve = Math.max(80, Math.min(290, horizontal * 0.5 + vertical * 0.2));

  const control1X = anchors.startX + curve * direction;
  const control2X = anchors.endX - curve * direction;

  return `M ${anchors.startX} ${anchors.startY} C ${control1X} ${anchors.startY}, ${control2X} ${anchors.endY}, ${anchors.endX} ${anchors.endY}`;
}

function createConnectionPath(className, d) {
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("class", className);
  path.setAttribute("d", d);
  path.setAttribute("marker-end", "url(#flow-arrow)");
  return path;
}

function deriveConnectionState(connection, runtimeNodes) {
  if (!runtimeNodes) {
    return "idle";
  }

  const sourceStatus = runtimeNodes[connection.from]?.status || "idle";
  const targetStatus = runtimeNodes[connection.to]?.status || "idle";
  const statuses = [sourceStatus, targetStatus];

  if (statuses.includes("fail")) {
    return "fail";
  }
  if (statuses.includes("running")) {
    return "running";
  }
  if (sourceStatus === "done" && targetStatus === "done") {
    return "done";
  }
  if (statuses.includes("done")) {
    return "running";
  }
  return "idle";
}

export function renderConnections({ lineGroup, connections, nodeMap, activeTypes, runtimeNodes }) {
  const fragment = document.createDocumentFragment();

  for (const connection of connections) {
    const source = nodeMap.get(connection.from);
    const target = nodeMap.get(connection.to);

    if (!source || !target) {
      continue;
    }

    const hidden =
      activeTypes &&
      (!activeTypes.has(source.type) || !activeTypes.has(target.type));
    const d = getBezierPath(source, target);
    const flowState = deriveConnectionState(connection, runtimeNodes);

    const echo = createConnectionPath("connection-line echo", d);
    const main = createConnectionPath("connection-line", d);
    echo.dataset.state = flowState;
    main.dataset.state = flowState;

    if (hidden) {
      echo.classList.add("is-hidden");
      main.classList.add("is-hidden");
    }

    echo.dataset.connectionId = connection.id;
    main.dataset.connectionId = connection.id;

    fragment.appendChild(echo);
    fragment.appendChild(main);
  }

  lineGroup.replaceChildren(fragment);
}
