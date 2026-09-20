import { CHAPTERS } from "./explainer-model.mjs";

const CHAPTER_COPY = [
  {
    title: "A useful proposal with one local error",
    takeaway: "A broadly correct action sequence can still fail because one local deviation disrupts execution.",
    input: "Frozen policy proposal A_base",
    change: "Inspect the highlighted near-future action",
    evidence: "Illustrative teaching values",
  },
  {
    title: "The proposal is an editable interface",
    takeaway: "The policy exposes a temporally structured sequence before physical execution.",
    input: "Ordered proposal tokens a0 through a7",
    change: "Preserve action identity while unfolding the sequence",
    evidence: "Method definition from the paper",
  },
  {
    title: "Predict a bounded residual, not a new policy",
    takeaway: "A compact refiner edits selected coordinates while the generating policy remains frozen.",
    input: "A_base with optional policy context",
    change: "Compose alpha times a masked residual with the proposal",
    evidence: "Illustrative teaching values",
  },
  {
    title: "Preserve the controller's native execution rule",
    takeaway: "ASRR changes the proposal while the base controller retains its original execution and replanning protocol.",
    input: "Refined proposal and native execution operator",
    change: "Consume the executable prefix, then replan",
    evidence: "Method contract from the paper",
  },
  {
    title: "The interface transfers across policy families",
    takeaway: "Recorded simulation and real-robot evaluations support the same output-space correction interface.",
    input: "Frozen policy proposals across five evidence cases",
    change: "Compare recorded Base and ASRR executions",
    evidence: "Recorded execution and measured aggregate",
  },
];

function point(action, key = "base") {
  const [x, y] = action[key];
  return [70 + x * 760, 54 + y * 350];
}

function trajectoryPath(actions, key) {
  return actions
    .map((action, index) => {
      const [x, y] = point(action, key);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function actionPoints(state, key, className) {
  return state.actions.map((action) => {
    const [x, y] = point(action, key);
    const selected = action.id === state.selectedActionId;
    return `<g class="action-point ${selected ? "is-selected" : ""}" data-trajectory-id="${action.id}" data-step="${action.index}" tabindex="0" role="button" aria-label="Select action ${action.id}">
      <circle class="${className}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${selected ? 10 : 6}"></circle>
      ${selected ? `<text x="${(x + 12).toFixed(1)}" y="${(y - 13).toFixed(1)}">${action.id}</text>` : ""}
    </g>`;
  }).join("");
}

function tokenStrip(state) {
  return `<div class="token-strip" aria-label="Predicted action sequence">
    ${state.actions.map((action) => `<button type="button" class="action-token ${action.id === state.selectedActionId ? "is-selected" : ""}" data-token-id="${action.id}" data-step="${action.index}" aria-pressed="${action.id === state.selectedActionId}"><span>${action.id}</span><small>t+${action.index}</small></button>`).join("")}
  </div>`;
}

function sceneFrame(state, content, label = "Illustrative") {
  return `<div class="scene" data-scene="${state.chapter.id}">
    <div class="scene-meta"><span>${label}</span><span>${String(state.selectedStep + 1).padStart(2, "0")} / 08 actions</span></div>
    ${content}
  </div>`;
}

function localErrorScene(state) {
  const errorAction = state.actions.find((action) => action.localError);
  const [x, y] = point(errorAction, "base");
  return sceneFrame(state, `<div class="diagram-wrap">
    <svg class="trajectory-svg" viewBox="0 0 900 455" role="img" aria-label="Illustrative action proposal with a local deviation">
      <path class="diagram-grid" d="M70 390H840M70 310H840M70 230H840M70 150H840M170 70V420M330 70V420M490 70V420M650 70V420M810 70V420"></path>
      <rect class="target-zone" x="775" y="72" width="70" height="70" rx="8"></rect>
      <text class="svg-label" x="782" y="62">target</text>
      <path class="trajectory trajectory--base" d="${trajectoryPath(state.actions, "base")}"></path>
      ${actionPoints(state, "base", "point--base")}
      <circle class="error-pulse" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="22"></circle>
      <path class="error-callout" d="M${(x + 16).toFixed(1)} ${(y - 17).toFixed(1)} L${(x + 92).toFixed(1)} ${(y - 74).toFixed(1)}"></path>
      <text class="svg-callout" x="${(x + 99).toFixed(1)}" y="${(y - 77).toFixed(1)}">local error</text>
    </svg>
  </div>
  ${tokenStrip(state)}
  <button class="recorded-evidence-chip" type="button" data-evidence-case="pi05-libero10">Recorded Base failure <span aria-hidden="true">&#8599;</span></button>`);
}

function editableProposalScene(state) {
  const reveal = Math.max(1, Math.ceil(state.chapterProgress * state.actions.length));
  return sceneFrame(state, `<div class="proposal-layout">
    <div class="frozen-policy-block"><span>Frozen policy</span><strong>f_base</strong><small>observation + instruction</small></div>
    <div class="proposal-arrow" aria-hidden="true">&#8594;</div>
    <div class="proposal-output"><span class="block-label">Structured output</span>${tokenStrip(state)}</div>
  </div>
  <svg class="identity-links" viewBox="0 0 900 230" aria-label="Action identities preserved from trajectory to tokens">
    <path class="trajectory trajectory--base" d="${trajectoryPath(state.actions, "base")}"></path>
    ${state.actions.map((action, index) => {
      const [x, y] = point(action, "base");
      return `<g data-trajectory-id="${action.id}" class="action-point ${action.id === state.selectedActionId ? "is-selected" : ""}" opacity="${index < reveal ? 1 : 0.18}"><circle class="point--base" cx="${x}" cy="${Math.max(28, y - 180)}" r="${action.id === state.selectedActionId ? 10 : 6}"></circle><text x="${x - 8}" y="${Math.max(20, y - 197)}">${action.id}</text></g>`;
    }).join("")}
  </svg>`, "Method interface");
}

function residualArrows(state) {
  return state.actions.slice(0, state.refinementHorizon).map((action) => {
    const [bx, by] = point(action, "base");
    const [rx, ry] = point(action, "refined");
    return `<path class="residual-arrow ${action.id === state.selectedActionId ? "is-selected" : ""}" d="M${bx} ${by} L${rx} ${ry}" marker-end="url(#arrowhead)" data-residual-id="${action.id}"></path>`;
  }).join("");
}

function residualScene(state) {
  const context = state.variant === "C" ? `<div class="context-block" data-context-path><span>Optional input</span><strong>Policy context</strong><small>global + step-aligned</small></div><div class="flow-arrow flow-arrow--context" data-context-path aria-hidden="true">&#8595;</div>` : "";
  return sceneFrame(state, `<div class="refiner-flow">
    <div class="flow-block flow-block--base"><span>Frozen proposal</span><strong>A<sub>base</sub></strong></div>
    <div class="flow-arrow" aria-hidden="true">&#8594;</div>
    <div class="flow-stack">${context}<div class="flow-block flow-block--refiner"><span>Compact sequence refiner</span><strong>R<sub>&theta;</sub></strong><small>${state.variant === "A" ? "action only" : "context conditioned"}</small></div></div>
    <div class="flow-arrow" aria-hidden="true">&#8594;</div>
    <div class="flow-block flow-block--residual"><span>Bounded residual</span><strong>&Delta;A</strong><small>mask M + scale &alpha;</small></div>
  </div>
  <div class="refinement-diagram">
    <svg class="trajectory-svg" viewBox="0 0 900 455" role="img" aria-label="Base proposal, residual vectors, and refined proposal">
      <defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="5" markerUnits="userSpaceOnUse" orient="auto"><polygon points="0 0, 10 5, 0 10"></polygon></marker></defs>
      <path class="diagram-grid" d="M70 390H840M70 310H840M70 230H840M70 150H840M170 70V420M330 70V420M490 70V420M650 70V420M810 70V420"></path>
      <path class="trajectory trajectory--base" d="${trajectoryPath(state.actions, "base")}"></path>
      <path class="trajectory trajectory--refined" d="${trajectoryPath(state.actions, "refined")}"></path>
      ${residualArrows(state)}
      ${actionPoints(state, state.before ? "base" : "refined", state.before ? "point--base" : "point--refined")}
    </svg>
    ${tokenStrip(state)}
  </div>`);
}

function nativeExecutionScene(state) {
  const progress = Math.min(state.refinementHorizon, Math.max(0, Math.floor(state.chapterProgress * (state.refinementHorizon + 1))));
  return sceneFrame(state, `<div class="execution-flow">
    <div class="flow-block flow-block--refined"><span>Edited proposal</span><strong>A<sub>refined</sub></strong></div>
    <div class="flow-arrow" aria-hidden="true">&#8594;</div>
    <div class="native-operator"><span>Base execution operator</span><strong>E<sub>base</sub></strong><small>native horizon + replanning rule</small></div>
    <div class="flow-arrow" aria-hidden="true">&#8594;</div>
    <div class="flow-block"><span>Robot command</span><strong>A<sub>exec</sub></strong></div>
  </div>
  <div class="horizon-panel">
    <div class="horizon-labels"><span>prediction horizon H<sub>p</sub></span><span>editable prefix H<sub>r</sub></span></div>
    <div class="execution-tokens">
      ${state.actions.map((action, index) => `<button type="button" class="execution-token ${index < state.refinementHorizon ? "is-editable" : "is-retained"} ${index < progress ? "is-executed" : ""}" data-token-id="${action.id}" data-trajectory-id="${action.id}" data-step="${index}" aria-label="Action ${action.id}, ${index < progress ? "executed" : index < state.refinementHorizon ? "editable" : "retained"}"><span>${action.id}</span><small>${index < progress ? "executed" : index < state.refinementHorizon ? "edited" : "base"}</small></button>`).join("")}
    </div>
    <div class="replan-marker" style="--replan-step:${progress}"><span>replan</span></div>
  </div>`, "Execution contract");
}

const EVIDENCE_CASES = [
  ["pi05-libero10", "pi0.5", "LIBERO-10", "pi05-base.png", "pi05-asrr.png"],
  ["openvla-goal", "OpenVLA-OFT", "LIBERO Goal", "openvla-base.png", "openvla-asrr.png"],
  ["corn", "Piper", "Corn to plate", "corn-base.png", "corn-asrr.png"],
  ["holder", "Piper", "Block to holder", "holder-base.png", "holder-asrr.png"],
  ["stack", "Piper", "Block stacking", "stack-base.png", "stack-asrr.png"],
];

function evidenceCard([id, policy, task, basePoster, asrrPoster], featured = false) {
  return `<button type="button" class="evidence-card ${featured ? "is-featured" : ""}" data-evidence-case="${id}">
    <span class="evidence-card__heading"><strong>${policy}</strong><small>${task}</small></span>
    <span class="evidence-thumbs"><span><img src="static/images/explainer/${basePoster}" alt="${task} Base rollout poster"><em>Base</em></span><span><img src="static/images/explainer/${asrrPoster}" alt="${task} ASRR rollout poster"><em>ASRR</em></span></span>
    <span class="evidence-card__action">Open recorded pair <span aria-hidden="true">&#8599;</span></span>
  </button>`;
}

function evidenceScene(state) {
  const expanded = state.chapterProgress >= 0.3;
  const cases = expanded ? EVIDENCE_CASES : [EVIDENCE_CASES[0]];
  return sceneFrame(state, `<div class="metric-ribbon" aria-label="Measured aggregate results">
    <div><strong>+14.4 pp</strong><span>pi0.5 mean gain</span></div>
    <div><strong>62.0 &#8594; 80.9</strong><span>5k checkpoint success</span></div>
    <div><strong>25-63%</strong><span>less recorded compute</span></div>
    <div><strong>+6.7 pp</strong><span>real-robot aggregate</span></div>
  </div>
  <div class="evidence-wall ${expanded ? "is-expanded" : "is-opening"}" data-evidence-wall>
    ${cases.map((item, index) => evidenceCard(item, !expanded && index === 0)).join("")}
  </div>`, "Recorded + measured");
}

function inspectorControls(state) {
  const refinement = state.chapterIndex === 2;
  return `<div id="variant-toggle" class="segmented-control ${refinement ? "" : "is-inactive"}" aria-label="Refiner input variant">
    <button type="button" data-variant="A" aria-pressed="${state.variant === "A"}" ${refinement ? "" : "disabled"}>ASRR-A</button>
    <button type="button" data-variant="C" aria-pressed="${state.variant === "C"}" ${refinement ? "" : "disabled"}>ASRR-C</button>
  </div>
  <button id="before-after" class="hold-command ${refinement ? "" : "is-inactive"}" type="button" aria-pressed="${state.before}" ${refinement ? "" : "disabled"}><span aria-hidden="true">&#8644;</span>Hold for Base proposal</button>`;
}

function renderInspector(inspectorElement, state) {
  const copy = CHAPTER_COPY[state.chapterIndex];
  inspectorElement.innerHTML = `<div class="inspector__header"><div><p class="overline">Current operation</p><span class="chapter-index">${String(state.chapterIndex + 1).padStart(2, "0")} / 05</span></div><span class="evidence-label ${state.chapterIndex === 4 ? "evidence-label--recorded" : "evidence-label--illustrative"}">${state.chapterIndex === 4 ? "Evidence" : "Method"}</span></div>
    <h2>${copy.title}</h2>
    <p class="inspector__takeaway"><strong>Takeaway.</strong> ${copy.takeaway}</p>
    <dl class="operation-list">
      <div><dt>Input</dt><dd>${copy.input}</dd></div>
      <div><dt>Change</dt><dd>${copy.change}; selected <code>${state.selectedActionId}</code></dd></div>
      <div><dt>Evidence</dt><dd>${copy.evidence}</dd></div>
    </dl>
    <div class="inspector__formula" aria-label="ASRR composition formula"><span>A<sub>refined</sub></span><span>=</span><span>A<sub>base</sub></span><span>+</span><span class="formula-residual">&alpha;(M &#8857; &Delta;A)</span></div>
    ${inspectorControls(state)}`;
}

export function renderScene(stageElement, inspectorElement, state) {
  const scenes = [localErrorScene, editableProposalScene, residualScene, nativeExecutionScene, evidenceScene];
  stageElement.innerHTML = scenes[state.chapterIndex](state);
  renderInspector(inspectorElement, state);
}

export function renderChapterTabs(container, state) {
  container.innerHTML = CHAPTERS.map((chapter) => `<button type="button" data-chapter-index="${chapter.index}" ${chapter.index === state.chapterIndex ? 'aria-current="step"' : ""}><span>${String(chapter.index + 1).padStart(2, "0")}</span>${chapter.label}</button>`).join("");
}

export function describeState(state) {
  const chapter = CHAPTERS[state.chapterIndex];
  return `${chapter.label}. ${CHAPTER_COPY[state.chapterIndex].takeaway} Selected action ${state.selectedActionId}.`;
}
