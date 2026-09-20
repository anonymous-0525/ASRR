import { CHAPTERS } from "./explainer-model.mjs";

const CHAPTER_COPY = [
  {
    title: "A useful proposal can still contain a local error",
    takeaway: "A local deviation at a5 sends the yellow Base proposal away from the blue ideal path, leaving a7 outside the target.",
    input: "Frozen policy proposal A_base",
    change: "Enter at a3 and follow the proposal through a7",
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

function trajectoryPath(actions, accessor) {
  return actions.map((action, index) => {
    const [x, y] = projectedPoint(accessor(action));
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

export function armGeometry([x, y], {
  origin = [345, 405],
  upper = 340,
  lower = 220,
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
  const asset = "static/images/explainer/arm";
  const link = (start, end, name) => {
    const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const angle = Math.atan2(end[1] - start[1], end[0] - start[0]) * 180 / Math.PI;
    return `<g transform="translate(${start[0]} ${start[1]}) rotate(${angle})">
      <image href="${asset}/${name}.png" x="${-length / 6}" y="${-length / 7.2}" width="${length * 4 / 3}" height="${length / 3.6}"></image>
    </g>`;
  };
  const joint = (center, size) => `<image href="${asset}/joint.png" x="${center[0] - size / 2}" y="${center[1] - size / 2}" width="${size}" height="${size}"></image>`;
  return `<g class="robot-arm ${className}" data-arm="${role}">
    <image href="${asset}/base.png" x="${arm.base[0] - 51.2}" y="${arm.base[1] - 32}" width="102.4" height="96"></image>
    ${link(arm.base, arm.elbow, "upper")}
    ${link(arm.elbow, arm.wrist, "forearm")}
    ${joint(arm.base, 66)}
    ${joint(arm.elbow, 55)}
    <image href="${asset}/gripper.png" x="${arm.wrist[0] - 24.7}" y="${arm.wrist[1] - 10.4}" width="49.4" height="46.8"></image>
    ${joint(arm.wrist, 28)}
    <circle class="robot-servo-accent" cx="${arm.elbow[0]}" cy="${arm.elbow[1]}" r="11"></circle>
    <circle class="robot-tool-center" cx="${arm.wrist[0]}" cy="${arm.wrist[1]}" r="5"></circle>
  </g>`;
}

function actionHistoryPoints(state, key = "base") {
  return state.actions.map((action) => {
    const [x, y] = point(action, key);
    const status = action.index === state.selectedStep
      ? "is-current"
      : action.index <= state.visitedThrough ? "is-visited" : "is-future";
    return `<g class="action-point ${key === "refined" ? "action-point--refined" : ""} ${status}" data-trajectory-id="${action.id}" data-action-label="${action.id}" data-step="${action.index}" tabindex="0" role="button" aria-label="Select action ${action.id}">
      <circle class="point--history" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${status === "is-current" ? 11 : 7}"></circle>
      <text x="${(x + 10).toFixed(1)}" y="${(y + (action.index >= 5 && key === "base" ? 24 : -15)).toFixed(1)}">${action.id}</text>
    </g>`;
  }).join("");
}

function targetRegion(state, refined = false) {
  const [x, y] = point(state.actions[7], "ideal");
  return `<g class="target-marker ${refined ? "target-marker--refined" : ""}">
    <rect class="target-zone" x="${x - 26}" y="${y - 23}" width="52" height="46" rx="5"></rect>
    <path class="target-cross" d="M${x - 7} ${y}h14M${x} ${y - 7}v14"></path>
    <text class="svg-label" text-anchor="middle" x="${x}" y="${y - 35}">target region</text>
  </g>`;
}

function trajectoryLegend(refined = false) {
  return `<div class="trajectory-legend" aria-label="Trajectory colors">
    <span class="legend-base">Base proposal</span>
    <span class="legend-ideal">Ideal path</span>
    ${refined ? '<span class="legend-refined">ASRR refined</span>' : ""}
  </div>`;
}

function sceneFrame(state, content, label = "Illustrative") {
  return `<div class="scene" data-scene="${state.chapter.id}">
    <div class="scene-meta"><span>${label}</span><span>${String(state.selectedStep + 1).padStart(2, "0")} / 08 actions</span></div>
    ${content}
  </div>`;
}

function localErrorScene(state) {
  const activePoint = projectedPoint(state.arm.base);
  const errorPoint = point(state.actions.find((action) => action.localError), "base");
  const errorReached = state.selectedStep >= 5;
  const [endX, endY] = point(state.actions[7]);
  return sceneFrame(state, `${trajectoryLegend()}<div class="diagram-wrap local-error-stage">
    <svg class="trajectory-svg" viewBox="0 0 900 500" role="img" aria-label="Yellow Base proposal deviates at a5 from the blue ideal path and misses the target at a7">
      <path class="diagram-grid" d="M90 460H840M90 355H840M90 250H840M90 145H840"></path>
      ${robotArm(projectedPoint(state.arm.ideal), "ideal-ghost", "robot-arm--ideal-ghost")}
      ${robotArm(activePoint, "active", "robot-arm--active")}
      ${targetRegion(state)}
      <path class="trajectory trajectory--ideal" data-path="ideal" d="${trajectoryPath(state.actions, (action) => action.ideal)}"></path>
      <path class="trajectory trajectory--history" d="${trajectoryPath(state.actions, (action) => action.base)}"></path>
      ${actionHistoryPoints(state)}
      <circle class="error-pulse ${errorReached ? "is-visible" : ""}" cx="${errorPoint[0].toFixed(1)}" cy="${errorPoint[1].toFixed(1)}" r="25"></circle>
      <g class="local-error-annotation ${errorReached ? "is-reached" : ""}">
        <path class="error-callout" d="M${errorPoint[0] + 20} ${errorPoint[1] + 18}Q610 292 644 295"></path>
        <text class="svg-callout" x="650" y="301">Local error at a5</text>
      </g>
      <g class="miss-annotation ${state.selectedStep === 7 ? "is-reached" : ""}">
        <path d="M${endX - 10} ${endY - 10}l20 20m0 -20l-20 20"></path>
        <text x="${endX + 27}" y="${endY + 6}">missed target</text>
      </g>
    </svg>
  </div>
  <button class="recorded-evidence-chip" type="button" data-evidence-case="pi05-libero10">Open a recorded failure <span aria-hidden="true">&#8599;</span></button>`);
}

function laneArm(state, key, role, className) {
  const refined = key === "refined";
  const [x, y] = projectedPoint(state.arm[key]);
  return `<svg class="interface-arm" viewBox="0 0 900 500" role="img" aria-label="${refined ? "ASRR corrects the Base proposal into a trajectory reaching the target" : "Base action trajectory with a local deviation"}">
    <path class="interface-ground" d="M220 460H510"></path>
    ${robotArm(projectedPoint(state.arm.ideal), "ideal-ghost", "robot-arm--ideal-ghost")}
    ${robotArm([x, y], role, className)}
    ${targetRegion(state, refined)}
    <path class="trajectory trajectory--ideal" data-path="ideal" d="${trajectoryPath(state.actions, (action) => action.ideal)}"></path>
    <path class="trajectory trajectory--history ${refined ? "trajectory--ghost" : ""}" d="${trajectoryPath(state.actions, (action) => action.base)}"></path>
    ${refined ? `<path class="trajectory trajectory--refined" d="${trajectoryPath(state.actions, (action) => action.refined)}"></path>` : ""}
    ${actionHistoryPoints(state, key)}
    <circle class="interface-action ${refined ? "interface-action--refined" : ""}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="10"></circle>
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
      <div class="lane-trajectory">${trajectoryLegend()}${laneArm(state, "base", "policy-output", "robot-arm--base")}</div>
    </section>
    <section class="adaptation-lane adaptation-lane--asrr">
      <div class="lane-heading"><span>ASRR</span><strong>Edit the proposal</strong></div>
      <div class="lane-flow lane-flow--asrr">
        <div class="mini-block mini-block--frozen">Frozen policy</div><span>&#8594;</span>
        <div class="mini-block mini-block--proposal">Editable action proposal</div><span>&#8594;</span>
        <div class="mini-block mini-block--refiner">Residual refiner</div>
      </div>
      <div class="lane-trajectory">${trajectoryLegend(true)}${laneArm(state, "refined", "editable-output", "robot-arm--refined")}</div>
    </section>
  </div>`, "Adaptation target");
}

function residualCurves(state, refinedValues) {
  return state.actions.slice(0, state.refinementHorizon).flatMap((action, index) => {
    if (action.residual.every((value) => value === 0)) return [];
    const [bx, by] = projectedPoint(action.base);
    const [rx, ry] = projectedPoint(refinedValues[index]);
    const cx = (bx + rx) / 2 + 11;
    const cy = (by + ry) / 2 - 18;
    return `<path class="residual-curve ${action.id === state.selectedActionId ? "is-current" : ""}" d="M${bx.toFixed(1)} ${by.toFixed(1)} Q${cx.toFixed(1)} ${cy.toFixed(1)} ${rx.toFixed(1)} ${ry.toFixed(1)}" marker-end="url(#residual-head)" data-residual-id="${action.id}"></path>`;
  }).join("");
}

function residualScene(state) {
  const refinedValues = state.actions.map((action) => action.refined);
  const basePoint = projectedPoint(state.arm.base);
  const refinedPoint = projectedPoint(state.arm.refined);
  // Attach the context callout to the midpoint of the a5 correction curve.
  const errorAction = state.actions.find((action) => action.localError);
  const errorBase = point(errorAction);
  const errorRefined = point(errorAction, "refined");
  const contextAnchor = [(errorBase[0] + errorRefined[0]) / 2 + 5.5, (errorBase[1] + errorRefined[1]) / 2 - 9];
  const context = state.variant === "C" ? `<div class="context-path" data-context-path>
    <div class="policy-data"><span>Policy-side data</span><strong>state + policy features</strong></div>
    <span class="context-link" aria-hidden="true">&#8594;</span>
  </div>` : "";
  return sceneFrame(state, `<div class="residual-mode-row residual-mode-row--${state.variant}">
    <div class="mode-pill is-active"><strong>ASRR-${state.variant}</strong><span>${state.variant === "A" ? "action proposal only" : "proposal + policy-side data"}</span></div>
    ${context}
    <div class="refiner-node"><span>Compact refiner</span><strong>&Delta;A</strong></div>
    <div class="method-status"><span>Base policy frozen</span><span>Native execution preserved</span></div>
  </div>
  ${trajectoryLegend(true)}
  <div class="refinement-diagram arm-comparison">
    <svg class="trajectory-svg" viewBox="0 0 900 500" role="img" aria-label="Ghost Base arm and Refined arm follow labelled trajectories; policy state guides the a5 residual in ASRR-C">
      <defs><marker id="residual-head" markerUnits="userSpaceOnUse" markerWidth="16" markerHeight="16" refX="14" refY="8" orient="auto"><polygon points="0 0, 16 8, 0 16"></polygon></marker></defs>
      <path class="diagram-grid" d="M90 460H840M90 355H840M90 250H840M90 145H840"></path>
      ${robotArm(basePoint, "base-ghost", "robot-arm--ghost")}
      ${robotArm(refinedPoint, "refined", "robot-arm--refined")}
      ${targetRegion(state, true)}
      <path class="trajectory trajectory--ideal" data-path="ideal" d="${trajectoryPath(state.actions, (action) => action.ideal)}"></path>
      <path class="trajectory trajectory--history trajectory--ghost" d="${trajectoryPath(state.actions, (action) => action.base)}"></path>
      <path class="trajectory trajectory--refined" d="${trajectoryPath(state.actions, (action) => action.refined)}"></path>
      ${state.before ? "" : residualCurves(state, refinedValues)}
      ${actionHistoryPoints(state)}
      <circle class="moving-refined-point" cx="${refinedPoint[0]}" cy="${refinedPoint[1]}" r="8"></circle>
      ${state.variant === "C" && !state.before ? `<g class="policy-state-annotation" data-policy-state-at="a5">
        <path class="policy-state-leader" d="M486 74C530 90 525 153 ${contextAnchor.join(" ")}"></path>
        <circle cx="${contextAnchor[0]}" cy="${contextAnchor[1]}" r="4"></circle>
        <rect x="164" y="12" width="338" height="68" rx="5"></rect>
        <text x="180" y="39">Policy state used at a5</text>
        <text class="policy-state-detail" x="180" y="62">conditions this residual correction</text>
      </g>` : ""}
    </svg>
    <div class="illustration-note">Illustrative action-space view. Residual displacement enlarged for clarity.</div>
  </div>`, "Method animation");
}

function evidenceScene(state, { homePath = "./" } = {}) {
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
      <nav class="evidence-links" aria-label="Explore the project">
        <a href="${homePath}#videos" data-page-section>View videos <span aria-hidden="true">&#8595;</span></a>
        <a href="${homePath}#results" data-page-section>View results <span aria-hidden="true">&#8595;</span></a>
      </nav>
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

export function renderScene(stageElement, inspectorElement, state, options = {}) {
  const scenes = [localErrorScene, editableInterfaceScene, residualScene, evidenceScene];
  stageElement.innerHTML = scenes[state.chapterIndex](state, options);
  renderInspector(inspectorElement, state);
}

export function renderChapterTabs(container, state) {
  container.innerHTML = CHAPTERS.map((chapter) => `<button type="button" data-chapter-index="${chapter.index}" ${chapter.index === state.chapterIndex ? 'aria-current="step"' : ""}><span>${String(chapter.index + 1).padStart(2, "0")}</span>${chapter.label}</button>`).join("");
}

export function describeState(state) {
  const chapter = CHAPTERS[state.chapterIndex];
  return `${chapter.label}. ${CHAPTER_COPY[state.chapterIndex].takeaway} Selected action ${state.selectedActionId}.`;
}
