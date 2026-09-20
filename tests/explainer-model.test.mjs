import test from "node:test";
import assert from "node:assert/strict";

import {
  TOUR_DURATION_MS,
  CHAPTERS,
  composeResidual,
  chapterAtTime,
  deriveTourState,
  validateTeachingState,
} from "../docs/static/js/explainer-model.mjs";

test("bounded masked residual edits only declared coordinates", () => {
  const refined = composeResidual(
    [0.2, -0.1, 0.5],
    [0.4, -0.3, 0.2],
    2,
    [1, 0, 1],
    [0.1, 0.1, 0.05],
  );

  assert.deepEqual(refined, [0.4, -0.1, 0.6]);
});

test("the redesigned tour has four equal 30-second chapters", () => {
  assert.equal(TOUR_DURATION_MS, 120_000);
  assert.deepEqual(CHAPTERS.map(({ startMs, endMs }) => [startMs, endMs]), [
    [0, 30_000],
    [30_000, 60_000],
    [60_000, 90_000],
    [90_000, 120_000],
  ]);
  assert.equal(chapterAtTime(30_000).id, "editable-interface");
  assert.equal(chapterAtTime(60_000).id, "residual-refinement");
  assert.equal(chapterAtTime(90_000).id, "recorded-evidence");
  assert.equal(chapterAtTime(120_000).id, "recorded-evidence");
});

test("tour boundaries always resolve to a valid chapter", () => {
  assert.equal(CHAPTERS.length, 4);
  assert.equal(chapterAtTime(-1).index, 0);
  assert.equal(chapterAtTime(120_000).index, 3);

  for (const chapter of CHAPTERS) {
    const state = deriveTourState(chapter.startMs);
    assert.equal(state.chapterIndex, chapter.index);
    assert.ok(state.chapterProgress >= 0 && state.chapterProgress <= 1);
  }
});

test("local error walks from a0 through a7 without skipping", () => {
  assert.equal(deriveTourState(0).selectedActionId, "a0");
  assert.equal(deriveTourState(29_999).selectedActionId, "a7");

  for (let index = 0; index < 8; index += 1) {
    const state = deriveTourState(index * 3_750 + 1);
    assert.equal(state.selectedStep, index);
    assert.equal(state.visitedThrough, index - 1);
  }
});

test("arbitrary seeks preserve stable action identities", () => {
  const times = [120_000, 12_000, 90_000, 60_000, 0, 105_000];
  const identities = times.map((timeMs) =>
    deriveTourState(timeMs).actions.map(({ id }) => id),
  );

  for (const ids of identities) {
    assert.deepEqual(ids, ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]);
  }
});

test("action-only and context-conditioned variants expose declared context", () => {
  const actionOnly = deriveTourState(70_000, { variant: "A" });
  const conditioned = deriveTourState(70_000, { variant: "C" });

  assert.equal(actionOnly.context, null);
  assert.ok(conditioned.context);
  assert.equal(conditioned.context.global.length, 2);
  assert.equal(conditioned.context.perStep.length, 8);
});

test("teaching state is valid throughout the authored tour", () => {
  for (let index = 0; index <= 100; index += 1) {
    const state = deriveTourState((TOUR_DURATION_MS * index) / 100);
    assert.equal(validateTeachingState(state), true);
  }
});
