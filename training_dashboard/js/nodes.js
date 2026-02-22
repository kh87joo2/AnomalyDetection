export const NODE_DIMENSIONS = {
  width: 220,
  height: 104
};

const TYPE_LABEL = {
  pi: "PI",
  function: "Function",
  agent: "Agent",
  tool: "Tool"
};

export function buildNodeMap(nodes) {
  return new Map(nodes.map((node) => [node.id, node]));
}

export function createNodeElement(node) {
  const article = document.createElement("article");
  article.className = "node-card";
  article.dataset.nodeId = node.id;
  article.dataset.type = node.type;

  article.innerHTML = `
    <div class="node-inner">
      <div class="node-icon">${node.icon || "N"}</div>
      <div>
        <h3 class="node-label">${node.label}</h3>
        <p class="node-subtitle">${node.subtitle || ""}</p>
        <span class="node-tag">${TYPE_LABEL[node.type] || "Node"}</span>
      </div>
    </div>
  `;

  setNodePosition(article, node.x, node.y);
  return article;
}

export function setNodePosition(element, x, y) {
  element.style.transform = `translate(${x}px, ${y}px)`;
}

export function renderNodes({ nodes, nodeLayer, activeTypes }) {
  const nodeElements = new Map();
  const fragment = document.createDocumentFragment();

  for (const node of nodes) {
    const element = createNodeElement(node);
    const hidden = activeTypes && !activeTypes.has(node.type);
    element.classList.toggle("is-hidden", Boolean(hidden));

    fragment.appendChild(element);
    nodeElements.set(node.id, element);
  }

  nodeLayer.replaceChildren(fragment);
  return nodeElements;
}

export function syncNodeVisibility({ nodes, nodeElements, activeTypes }) {
  for (const node of nodes) {
    const element = nodeElements.get(node.id);
    if (!element) {
      continue;
    }

    const hidden = activeTypes && !activeTypes.has(node.type);
    element.classList.toggle("is-hidden", Boolean(hidden));
  }
}
