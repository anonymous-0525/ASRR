export const TOUR_DURATION_MS = 90_000;

export const CHAPTERS = Object.freeze([
  { index: 0, id: "local-error", label: "Local Error", startMs: 0, endMs: 16_000 },
  { index: 1, id: "editable-proposal", label: "Editable Proposal", startMs: 16_000, endMs: 31_000 },
  { index: 2, id: "residual-refinement", label: "Residual Refinement", startMs: 31_000, endMs: 55_000 },
  { index: 3, id: "native-execution", label: "Native Execution", startMs: 55_000, endMs: 70_000 },
  { index: 4, id: "evidence", label: "Evidence", startMs: 70_000, endMs: 90_000 },
]);

const BASE_POINTS = [
  [0.08, 0.78],
  [0.19, 0.69],
  [0.31, 0.59],
  [0.43, 0.49],
  [0.55, 0.4],
  [0.68, 0.38],
  [0.8, 0.24],
  [0.91, 0.17],
];

const RESIDUALS = [
  [0.0, 0.0],
  [0.0, 0.0],
  [0.005, -0.004],
  [0.008, -0.012],
  [0.014, -0.026],
  [0.022, -0.075],
  [0.012, -0.045],
  [0.004, -0.012],
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
      residual: freezePoint(RESIDUALS[index]),
      mask: freezePoint([1, 1]),
      bound: freezePoint([0.08, 0.08]),
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

function authoredStep(chapterIndex, chapterProgress) {
  if (chapterIndex === 0) return chapterProgress < 0.42 ? 3 : 5;
  if (chapterIndex === 1) return Math.min(7, Math.floor(chapterProgress * 8));
  if (chapterIndex === 2) return chapterProgress < 0.38 ? 4 : 5;
  if (chapterIndex === 3) return Math.min(7, Math.floor(chapterProgress * 6));
  return 5;
}

function authoredVariant(chapterIndex, chapterProgress) {
  return chapterIndex === 2 && chapterProgress < 0.34 ? "A" : "C";
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
  const selectedStep = Number.isInteger(overrides.selectedStep)
    ? clamp(overrides.selectedStep, 0, TEACHING_SEQUENCE.length - 1)
    : authoredStep(chapter.index, chapterProgress);
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

  return {
    timeMs: clampedTime,
    tourProgress: clampedTime / TOUR_DURATION_MS,
    chapter,
    chapterIndex: chapter.index,
    chapterProgress,
    variant,
    selectedStep,
    selectedActionId: actions[selectedStep].id,
    before,
    alpha,
    predictionHorizon: actions.length,
    refinementHorizon: 6,
    actions,
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
  if (state.selectedActionId !== state.actions[state.selectedStep].id) return false;
  if (state.variant === "A" && state.context !== null) return false;
  if (state.variant === "C" && !state.context) return false;
  return true;
}
