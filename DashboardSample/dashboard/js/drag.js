const MIN_ZOOM = 0.5;
const MAX_ZOOM = 1.9;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function setupInteractions({
  canvasWrapper,
  getTransform,
  setTransform,
  getNodeById,
  onNodeDrag,
  onTransform,
  setStatus
}) {
  let nodeDrag = null;
  let panDrag = null;

  const toWorldPoint = (event, transform) => {
    const rect = canvasWrapper.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - transform.x) / transform.scale,
      y: (event.clientY - rect.top - transform.y) / transform.scale
    };
  };

  const beginNodeDrag = (event, nodeElement) => {
    const nodeId = nodeElement.dataset.nodeId;
    const node = getNodeById(nodeId);

    if (!node) {
      return;
    }

    const transform = getTransform();
    const pointerWorld = toWorldPoint(event, transform);

    nodeDrag = {
      pointerId: event.pointerId,
      nodeId,
      offsetX: pointerWorld.x - node.x,
      offsetY: pointerWorld.y - node.y
    };

    nodeElement.classList.add("dragging");
    canvasWrapper.setPointerCapture(event.pointerId);
    setStatus(`Dragging ${node.label}`);
  };

  const beginPanDrag = (event) => {
    const transform = getTransform();
    panDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: transform.x,
      originY: transform.y
    };

    canvasWrapper.setPointerCapture(event.pointerId);
    setStatus("Panning canvas");
  };

  const endPointerAction = (event) => {
    if (nodeDrag && event.pointerId === nodeDrag.pointerId) {
      const draggingElement = canvasWrapper.querySelector(
        `.node-card[data-node-id="${nodeDrag.nodeId}"]`
      );
      if (draggingElement) {
        draggingElement.classList.remove("dragging");
      }

      nodeDrag = null;
      setStatus("Node drag complete");
    }

    if (panDrag && event.pointerId === panDrag.pointerId) {
      panDrag = null;
      setStatus("Pan complete");
    }

    if (canvasWrapper.hasPointerCapture(event.pointerId)) {
      canvasWrapper.releasePointerCapture(event.pointerId);
    }
  };

  canvasWrapper.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.pointerType !== "touch") {
      return;
    }

    const nodeElement = event.target.closest(".node-card");
    if (nodeElement) {
      event.preventDefault();
      beginNodeDrag(event, nodeElement);
      return;
    }

    beginPanDrag(event);
  });

  canvasWrapper.addEventListener("pointermove", (event) => {
    if (nodeDrag && event.pointerId === nodeDrag.pointerId) {
      event.preventDefault();
      const transform = getTransform();
      const pointerWorld = toWorldPoint(event, transform);
      const nextX = pointerWorld.x - nodeDrag.offsetX;
      const nextY = pointerWorld.y - nodeDrag.offsetY;

      onNodeDrag(nodeDrag.nodeId, nextX, nextY);
      return;
    }

    if (panDrag && event.pointerId === panDrag.pointerId) {
      event.preventDefault();
      const dx = event.clientX - panDrag.startX;
      const dy = event.clientY - panDrag.startY;
      setTransform({
        x: panDrag.originX + dx,
        y: panDrag.originY + dy,
        scale: getTransform().scale
      });
      onTransform();
    }
  });

  canvasWrapper.addEventListener("pointerup", endPointerAction);
  canvasWrapper.addEventListener("pointercancel", endPointerAction);
  canvasWrapper.addEventListener("lostpointercapture", endPointerAction);

  canvasWrapper.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();

      const current = getTransform();
      const rect = canvasWrapper.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      const worldX = (localX - current.x) / current.scale;
      const worldY = (localY - current.y) / current.scale;
      const nextScale = clamp(
        current.scale * Math.exp(-event.deltaY * 0.0014),
        MIN_ZOOM,
        MAX_ZOOM
      );

      setTransform({
        x: localX - worldX * nextScale,
        y: localY - worldY * nextScale,
        scale: nextScale
      });
      onTransform();
      setStatus(`Zoom ${Math.round(nextScale * 100)}%`);
    },
    { passive: false }
  );
}
