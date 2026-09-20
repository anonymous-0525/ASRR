export const TOUR_DURATION_MS = 105_000;

export const CHAPTERS = Object.freeze([
  { index: 0, id: "local-error", label: "Local Error", startMs: 0, endMs: 25_000 },
  { index: 1, id: "editable-interface", label: "Editable Interface", startMs: 25_000, endMs: 50_000 },
  { index: 2, id: "residual-refinement", label: "Residual Refinement", startMs: 50_000, endMs: 75_000 },
  { index: 3, id: "recorded-evidence", label: "Recorded Evidence", startMs: 75_000, endMs: 105_000 },
]);

const BASE_POINTS = [
  [0.08, 0.78],
  [0.19, 0.67],
  [0.31, 0.54],
  [0.43, 0.42],
  [0.55, 0.33],
  [0.65, 0.53],
  [0.79, 0.43],
  [0.91, 0.38],
];

const IDEAL_POINTS = [
  ...BASE_POINTS.slice(0, 5),
  [0.65, 0.22],
  [0.79, 0.15],
  [0.91, 0.13],
];

const STEP_CONTEXT = [0.12, 0.17, 0.24, 0.36, 0.58, 0.91, 0.54, 0.2];

function freezePoint(point) {
  return Object.freeze([...point]);
}

export const TEACHING_SEQUENCE = Object.freeze(
  BASE_POINTS.map((base, index) =>
    Object.freeze({
      id: `a${index}`,
      index,
      base: freezePoint(base),
      ideal: freezePoint(IDEAL_POINTS[index]),
      residual: freezePoint(IDEAL_POINTS[index].map((value, coordinate) => value - base[coordinate])),
      mask: freezePoint([1, 1]),
      bound: freezePoint([0.35, 0.35]),
      context: STEP_CONTEXT[index],
      localError: index === 5,
    }),
  ),
);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampTime(timeMs) {
  return clamp(Number.isFinite(timeMs) ? timeMs : 0, 0, TOUR_DURATION_MS);
}

export function composeResidual(base, delta, alpha = 1, mask, bound) {
  if (![base, delta, mask, bound].every(Array.isArray)) {
    throw new TypeError("Residual composition requires array inputs.");
  }
  if (![delta, mask, bound].every((values) => values.length === base.length)) {
    throw new RangeError("Residual inputs must use the same action dimension.");
  }

  return base.map((value, coordinate) => {
    const limited = clamp(delta[coordinate], -Math.abs(bound[coordinate]), Math.abs(bound[coordinate]));
    return value + alpha * mask[coordinate] * limited;
  });
}

export function chapterAtTime(timeMs) {
  const clamped = clampTime(timeMs);
  if (clamped === TOUR_DURATION_MS) {
    return CHAPTERS[CHAPTERS.length - 1];
  }
  return CHAPTERS.find(
    (chapter) => clamped >= chapter.startMs && clamped < chapter.endMs,
  ) ?? CHAPTERS[0];
}

function authoredVariant(chapterIndex, chapterProgress) {
  return chapterIndex === 2 && chapterProgress < 0.5 ? "A" : "C";
}

function sequenceProgress(chapterIndex, chapterProgress) {
  if (chapterIndex !== 2) return chapterProgress;
  return chapterProgress < 0.5
    ? chapterProgress * 2
    : (chapterProgress - 0.5) * 2;
}

function interpolateAction(actions, position, key) {
  const fromIndex = Math.floor(position);
  const toIndex = Math.min(actions.length - 1, fromIndex + 1);
  const blend = position - fromIndex;
  return actions[fromIndex][key].map((value, coordinate) =>
    value + (actions[toIndex][key][coordinate] - value) * blend);
}

export function deriveTourState(timeMs, overrides = {}) {
  const clampedTime = clampTime(timeMs);
  const chapter = chapterAtTime(clampedTime);
  const duration = chapter.endMs - chapter.startMs;
  const chapterProgress = clampedTime === TOUR_DURATION_MS
    ? 1
    : clamp((clampedTime - chapter.startMs) / duration, 0, 1);
  const variant = overrides.variant === "A" || overrides.variant === "C"
    ? overrides.variant
    : authoredVariant(chapter.index, chapterProgress);
  const firstStep = chapter.index === 0 ? 3 : 0;
  const motionProgress = clamp(sequenceProgress(chapter.index, chapterProgress) / 0.8, 0, 1);
  const actionPosition = Number.isInteger(overrides.selectedStep)
    ? clamp(overrides.selectedStep, 0, TEACHING_SEQUENCE.length - 1)
    : firstStep + motionProgress * (TEACHING_SEQUENCE.length - 1 - firstStep);
  const selectedStep = Number.isInteger(overrides.selectedStep)
    ? clamp(overrides.selectedStep, 0, TEACHING_SEQUENCE.length - 1)
    : chapter.index === 3 ? 5 : Math.floor(actionPosition);
  const visitedThrough = selectedStep - 1;
  const before = typeof overrides.before === "boolean" ? overrides.before : false;
  const alpha = 1;

  const actions = TEACHING_SEQUENCE.map((action) => ({
    ...action,
    base: [...action.base],
    residual: [...action.residual],
    refined: before
      ? [...action.base]
      : composeResidual(action.base, action.residual, alpha, action.mask, action.bound),
  }));
  const evidenceOffsetMs = Math.max(0, clampedTime - CHAPTERS[3].startMs);
  const evidenceCaseId = chapter.index === 3 && evidenceOffsetMs >= 15_000
    ? "corn"
    : "pi05-libero10";
  const evidenceLocalTimeMs = chapter.index === 3
    ? Math.min(15_000, evidenceOffsetMs - (evidenceCaseId === "corn" ? 15_000 : 0))
    : 0;

  return {
    timeMs: clampedTime,
    tourProgress: clampedTime / TOUR_DURATION_MS,
    chapter,
    chapterIndex: chapter.index,
    chapterProgress,
    variant,
    selectedStep,
    actionPosition,
    visitedThrough,
    selectedActionId: actions[selectedStep].id,
    before,
    alpha,
    predictionHorizon: actions.length,
    refinementHorizon: actions.length,
    actions,
    arm: {
      base: interpolateAction(actions, actionPosition, "base"),
      refined: interpolateAction(actions, actionPosition, "refined"),
    },
    evidenceCaseId,
    evidenceLocalTimeMs,
    context: variant === "C"
      ? {
          global: [0.64, 0.28],
          perStep: actions.map((action) => action.context),
        }
      : null,
  };
}

export function validateTeachingState(state) {
  if (!state || !CHAPTERS[state.chapterIndex]) return false;
  if (state.timeMs < 0 || state.timeMs > TOUR_DURATION_MS) return false;
  if (state.chapterProgress < 0 || state.chapterProgress > 1) return false;
  if (!Array.isArray(state.actions) || state.actions.length !== 8) return false;
  if (state.actions.some((action, index) => action.id !== `a${index}`)) return false;
  if (!Number.isInteger(state.selectedStep) || !state.actions[state.selectedStep]) return false;
  if (!Number.isInteger(state.visitedThrough) || state.visitedThrough < -1 || state.visitedThrough > 7) return false;
  if (state.selectedActionId !== state.actions[state.selectedStep].id) return false;
  if (!state.arm || state.arm.base.length !== 2 || state.arm.refined.length !== 2) return false;
  if (state.variant === "A" && state.context !== null) return false;
  if (state.variant === "C" && !state.context) return false;
  return true;
}
