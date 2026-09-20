import { CHAPTERS } from "./explainer-model.mjs";

const CHAPTER_COPY = [
  {
    title: "A useful proposal can still contain a local error",
    takeaway: "The arm follows a broadly useful action proposal until one local deviation disrupts execution.",
    input: "Frozen policy proposal A_base",
    change: "Follow each action from a0 through a7",
    evidence: "Illustrative action-space projection",
  },
  {
    title: "Make the proposal the editable interface",
    takeaway: "ASRR keeps the base policy fixed and redirects adaptation to the action it already produced.",
    input: "Observation, instruction, and Base action",
    change: "Compare policy updates with proposal editing",
    evidence: "Method interface from the paper",
  },
  {
    title: "Refine actions alone or add policy-side data",
    takeaway: "A compact refiner predicts bounded local edits from the proposal, optionally conditioned on policy-side information.",
    input: "A_base, optionally with state, mode, or generation features",
    change: "Apply a masked residual at each active action",
    evidence: "Illustrative geometry, displacement enlarged for clarity",
  },
  {
    title: "Compare recorded Base and ASRR executions",
    takeaway: "Paired simulation and physical executions show the same action-editing interface beyond the schematic.",
    input: "Recorded pi0.5 and Piper executions",
    change: "Play Base failure beside ASRR success",
    evidence: "Recorded execution and measured aggregate",
  },
];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function point(action, key = "base") {
  const [x, y] = action[key];
  return [70 + x * 760, 54 + y * 350];
}

function projectedPoint(values) {
  return [70 + values[0] * 760, 54 + values[1] * 350];
}

function illustrativeRefined(action) {
  const scale = action.localError ? 3.6 : 2.45;
  return [
    action.base[0] + action.residual[0] * scale,
    action.base[1] + action.residual[1] * scale,
  ];
}

function trajectoryPath(actions, accessor) {
  return actions.map((action, index) => {
    const [x, y] = projectedPoint(accessor(action));
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

export function armGeometry([x, y], {
  origin = [155, 386],
  upper = 335,
  lower = 335,
} = {}) {
  const dx = x - origin[0];
  const dy = y - origin[1];
  const distance = Math.hypot(dx, dy);
  const reach = clamp(distance, Math.abs(upper - lower) + 1, upper + lower - 1);
  const heading = Math.atan2(dy, dx);
  const cosine = clamp(
    (upper ** 2 + reach ** 2 - lower ** 2) / (2 * upper * reach),
    -1,
    1,
  );
  const elbowOffset = Math.acos(cosine);
  const elbow = [
    origin[0] + upper * Math.cos(heading + elbowOffset),
    origin[1] + upper * Math.sin(heading + elbowOffset),
  ];
  const gripper = [
    [x + 13 * Math.cos(heading - 0.52), y + 13 * Math.sin(heading - 0.52)],
    [x + 13 * Math.cos(heading + 0.52), y + 13 * Math.sin(heading + 0.52)],
  ];
  return { base: [...origin], elbow, wrist: [x, y], gripper };
}

function robotArm(target, role, className = "") {
  const arm = armGeometry(target);
  return `<g class="robot-arm ${className}" data-arm="${role}">
    <line class="robot-link robot-link--upper" x1="${arm.base[0]}" y1="${arm.base[1]}" x2="${arm.elbow[0].toFixed(1)}" y2="${arm.elbow[1].toFixed(1)}"></line>
    <line class="robot-link robot-link--lower" x1="${arm.elbow[0].toFixed(1)}" y1="${arm.elbow[1].toFixed(1)}" x2="${arm.wrist[0]}" y2="${arm.wrist[1]}"></line>
    <circle class="robot-base" cx="${arm.base[0]}" cy="${arm.base[1]}" r="18"></circle>
    <circle class="robot-joint" cx="${arm.elbow[0].toFixed(1)}" cy="${arm.elbow[1].toFixed(1)}" r="11"></circle>
    <circle class="robot-wrist" cx="${arm.wrist[0]}" cy="${arm.wrist[1]}" r="9"></circle>
    <line class="robot-gripper" x1="${arm.wrist[0]}" y1="${arm.wrist[1]}" x2="${arm.gripper[0][0].toFixed(1)}" y2="${arm.gripper[0][1].toFixed(1)}"></line>
    <line class="robot-gripper" x1="${arm.wrist[0]}" y1="${arm.wrist[1]}" x2="${arm.gripper[1][0].toFixed(1)}" y2="${arm.gripper[1][1].toFixed(1)}"></line>
  </g>`;
}

function actionHistoryPoints(state) {
  return state.actions.map((action) => {
    const [x, y] = point(action, "base");
    const status = action.index === state.selectedStep
      ? "is-current"
      : action.index <= state.visitedThrough ? "is-visited" : "is-future";
    return `<g class="action-point ${status}" data-trajectory-id="${action.id}" data-action-label="${action.id}" data-step="${action.index}" tabindex="0" role="button" aria-label="Select action ${action.id}">
      <circle class="point--history" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${status === "is-current" ? 11 : 7}"></circle>
      <text x="${(x + 11).toFixed(1)}" y="${(y - 13).toFixed(1)}">${action.id}</text>
    </g>`;
  }).join("");
}

function sceneFrame(state, content, label = "Illustrative") {
  return `<div class="scene" data-scene="${state.chapter.id}">
    <div class="scene-meta"><span>${label}</span><span>${String(state.selectedStep + 1).padStart(2, "0")} / 08 actions</span></div>
    ${content}
  </div>`;
}

function localErrorScene(state) {
  const active = state.actions[state.selectedStep];
  const activePoint = projectedPoint(state.arm.base);
  const errorPoint = point(state.actions.find((action) => action.localError), "base");
  const errorReached = state.selectedStep >= 5;
  return sceneFrame(state, `<div class="diagram-wrap local-error-stage">
    <svg class="trajectory-svg" viewBox="0 0 900 455" role="img" aria-label="Robot arm following eight labelled actions toward one local error">
      <path class="diagram-grid" d="M70 390H840M70 310H840M70 230H840M70 150H840M170 70V420M330 70V420M490 70V420M650 70V420M810 70V420"></path>
      <rect class="target-zone" x="774" y="72" width="72" height="70" rx="8"></rect>
      <text class="svg-label" x="780" y="62">target region</text>
      <path class="trajectory trajectory--history" d="${trajectoryPath(state.actions, (action) => action.base)}"></path>
      ${actionHistoryPoints(state)}
      ${robotArm(activePoint, "active", "robot-arm--active")}
      <circle class="error-pulse ${errorReached ? "is-visible" : ""}" cx="${errorPoint[0].toFixed(1)}" cy="${errorPoint[1].toFixed(1)}" r="25"></circle>
      <path class="error-callout ${errorReached ? "is-visible" : ""}" d="M${(errorPoint[0] + 18).toFixed(1)} ${(errorPoint[1] - 18).toFixed(1)} Q${(errorPoint[0] + 62).toFixed(1)} ${(errorPoint[1] - 76).toFixed(1)} ${(errorPoint[0] + 128).toFixed(1)} ${(errorPoint[1] - 82).toFixed(1)}"></path>
      <text class="svg-callout ${errorReached ? "is-visible" : ""}" x="${(errorPoint[0] + 134).toFixed(1)}" y="${(errorPoint[1] - 84).toFixed(1)}">local error</text>
    </svg>
  </div>
  <button class="recorded-evidence-chip" type="button" data-evidence-case="pi05-libero10">Open a recorded failure <span aria-hidden="true">&#8599;</span></button>`);
}

function laneArm(state, key, role, className) {
  const action = state.actions[state.selectedStep];
  const scale = action.localError ? 3.6 : 2.45;
  const values = key === "refined"
    ? state.arm.base.map((value, coordinate) => value + action.residual[coordinate] * scale)
    : state.arm.base;
  const [x, y] = projectedPoint(values);
  return `<svg class="interface-arm" viewBox="0 0 900 455" aria-hidden="true">
    <path class="interface-ground" d="M75 400H845"></path>
    ${robotArm([x, y], role, className)}
    <circle class="interface-action ${key === "refined" ? "interface-action--refined" : ""}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="12"></circle>
  </svg>`;
}

function editableInterfaceScene(state) {
  return sceneFrame(state, `<div class="adaptation-compare">
    <section class="adaptation-lane adaptation-lane--policy">
      <div class="lane-heading"><span>Conventional adaptation</span><strong>Adapt the policy</strong></div>
      <div class="lane-flow">
        <div class="mini-block">Observation + instruction</div><span>&#8594;</span>
        <div class="mini-block mini-block--policy">Policy parameters<small>additional optimization</small></div><span>&#8594;</span>
        <div class="mini-block">Robot action</div>
      </div>
      ${laneArm(state, "base", "policy-output", "robot-arm--base")}
    </section>
    <section class="adaptation-lane adaptation-lane--asrr">
      <div class="lane-heading"><span>ASRR</span><strong>Edit the proposal</strong></div>
      <div class="lane-flow lane-flow--asrr">
        <div class="mini-block mini-block--frozen">Frozen policy</div><span>&#8594;</span>
        <div class="mini-block mini-block--proposal">Editable action proposal</div><span>&#8594;</span>
        <div class="mini-block mini-block--refiner">Residual refiner</div>
      </div>
      ${laneArm(state, "refined", "editable-output", "robot-arm--refined")}
    </section>
  </div>`, "Adaptation target");
}

function residualCurves(state, refinedValues) {
  return state.actions.slice(0, state.refinementHorizon).map((action, index) => {
    const [bx, by] = projectedPoint(action.base);
    const [rx, ry] = projectedPoint(refinedValues[index]);
    const cx = (bx + rx) / 2 + 11;
    const cy = (by + ry) / 2 - 18;
    return `<path class="residual-curve ${action.id === state.selectedActionId ? "is-current" : ""}" d="M${bx.toFixed(1)} ${by.toFixed(1)} Q${cx.toFixed(1)} ${cy.toFixed(1)} ${rx.toFixed(1)} ${ry.toFixed(1)}" marker-end="url(#residual-head)" data-residual-id="${action.id}"></path>`;
  }).join("");
}

function residualScene(state) {
  const refinedValues = state.actions.map(illustrativeRefined);
  const active = state.actions[state.selectedStep];
  const basePoint = projectedPoint(state.arm.base);
  const scale = active.localError ? 3.6 : 2.45;
  const movingRefined = state.arm.base.map((value, coordinate) =>
    value + active.residual[coordinate] * scale);
  const refinedPoint = projectedPoint(state.before ? state.arm.base : movingRefined);
  const context = state.variant === "C" ? `<div class="policy-data" data-context-path>
    <span>Policy-side data</span><strong>state + mode + generation features</strong>
  </div><div class="context-link" data-context-path aria-hidden="true">&#8600;</div>` : "";
  return sceneFrame(state, `<div class="residual-mode-row">
    <div class="mode-pill is-active"><strong>ASRR-${state.variant}</strong><span>${state.variant === "A" ? "action proposal only" : "proposal + policy-side data"}</span></div>
    ${context}
    <div class="refiner-node"><span>Compact refiner</span><strong>&Delta;A</strong></div>
    <div class="method-status"><span>Base policy frozen</span><span>Native execution preserved</span></div>
  </div>
  <div class="refinement-diagram arm-comparison">
    <svg class="trajectory-svg" viewBox="0 0 900 455" role="img" aria-label="Ghost Base arm and opaque Refined arm with curved residual corrections">
      <defs><marker id="residual-head" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><polygon points="0 0, 9 4.5, 0 9"></polygon></marker></defs>
      <path class="diagram-grid" d="M70 390H840M70 310H840M70 230H840M70 150H840M170 70V420M330 70V420M490 70V420M650 70V420M810 70V420"></path>
      <path class="trajectory trajectory--history trajectory--ghost" d="${trajectoryPath(state.actions, (action) => action.base)}"></path>
      <path class="trajectory trajectory--refined" d="${trajectoryPath(state.actions, (action) => state.before ? action.base : illustrativeRefined(action))}"></path>
      ${state.before ? "" : residualCurves(state, refinedValues)}
      ${robotArm(basePoint, "base-ghost", "robot-arm--ghost")}
      ${robotArm(refinedPoint, "refined", "robot-arm--refined")}
      <text class="arm-label arm-label--base" x="95" y="80">Base arm</text>
      <text class="arm-label arm-label--refined" x="95" y="105">Refined arm</text>
    </svg>
    <div class="illustration-note">Illustrative action-space view. Residual displacement enlarged for clarity.</div>
  </div>`, "Method animation");
}

function evidenceScene(state) {
  const realRobot = state.evidenceCaseId === "corn";
  return sceneFrame(state, `<div class="authored-evidence" data-authored-evidence="${state.evidenceCaseId}">
    <div class="authored-evidence__heading">
      <div><span>${realRobot ? "Physical manipulation" : "VLA simulation"}</span><strong>${realRobot ? "Piper / Corn-to-plate" : "pi0.5 / LIBERO-10"}</strong></div>
      <span class="phase-badge">${realRobot ? "3x playback" : "Recorded rollout"}</span>
    </div>
    <div class="authored-media-pair" data-authored-media-pair aria-label="Synchronized Base and ASRR recorded executions"></div>
    <div class="evidence-overlay" aria-label="Measured aggregate results">
      ${realRobot
        ? '<div><strong>+6.7 pp</strong><span>real-robot aggregate</span></div><div><strong>170 / 240</strong><span>ASRR successes</span></div>'
        : '<div><strong>+14.4 pp</strong><span>pi0.5 mean gain</span></div><div><strong>25-63%</strong><span>less recorded compute</span></div>'}
      <button type="button" data-open-evidence-library>More evidence <span aria-hidden="true">&#8599;</span></button>
    </div>
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
  inspectorElement.innerHTML = `<div class="inspector__header"><div><p class="overline">Current operation</p><span class="chapter-index">${String(state.chapterIndex + 1).padStart(2, "0")} / 04</span></div><span class="evidence-label ${state.chapterIndex === 3 ? "evidence-label--recorded" : "evidence-label--illustrative"}">${state.chapterIndex === 3 ? "Evidence" : "Method"}</span></div>
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
  const scenes = [localErrorScene, editableInterfaceScene, residualScene, evidenceScene];
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
