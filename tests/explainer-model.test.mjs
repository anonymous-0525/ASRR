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

test("tour boundaries always resolve to a valid chapter", () => {
  assert.equal(TOUR_DURATION_MS, 90_000);
  assert.equal(CHAPTERS.length, 5);
  assert.equal(chapterAtTime(-1).index, 0);
  assert.equal(chapterAtTime(90_000).index, 4);

  for (const chapter of CHAPTERS) {
    const state = deriveTourState(chapter.startMs);
    assert.equal(state.chapterIndex, chapter.index);
    assert.ok(state.chapterProgress >= 0 && state.chapterProgress <= 1);
  }
});

test("arbitrary seeks preserve stable action identities", () => {
  const times = [90_000, 12_000, 55_000, 31_000, 0, 72_500];
  const identities = times.map((timeMs) =>
    deriveTourState(timeMs).actions.map(({ id }) => id),
  );

  for (const ids of identities) {
    assert.deepEqual(ids, ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]);
  }
});

test("action-only and context-conditioned variants expose declared context", () => {
  const actionOnly = deriveTourState(43_000, { variant: "A" });
  const conditioned = deriveTourState(43_000, { variant: "C" });

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
