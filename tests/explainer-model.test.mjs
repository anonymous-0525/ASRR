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

test("the tour speeds up the first three chapters and preserves 30 seconds of evidence", () => {
  assert.equal(TOUR_DURATION_MS, 105_000);
  assert.deepEqual(CHAPTERS.map(({ startMs, endMs }) => [startMs, endMs]), [
    [0, 25_000],
    [25_000, 50_000],
    [50_000, 75_000],
    [75_000, 105_000],
  ]);
  assert.equal(chapterAtTime(25_000).id, "editable-interface");
  assert.equal(chapterAtTime(50_000).id, "residual-refinement");
  assert.equal(chapterAtTime(75_000).id, "recorded-evidence");
  assert.equal(chapterAtTime(105_000).id, "recorded-evidence");
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

test("local error begins at a3 and walks through a7 without skipping", () => {
  assert.equal(deriveTourState(0).selectedActionId, "a3");
  assert.equal(deriveTourState(24_999).selectedActionId, "a7");

  for (let index = 3; index < 8; index += 1) {
    const state = deriveTourState((index - 3) * 5_000 + 1);
    assert.equal(state.selectedStep, index);
    assert.equal(state.visitedThrough, index - 1);
  }
});

test("the robot arm interpolates continuously between authored actions", () => {
  const state = deriveTourState(1_875);
  const a3 = state.actions[3].base;
  const a4 = state.actions[4].base;

  assert.ok(state.arm.base[0] > a3[0] && state.arm.base[0] < a4[0]);
  assert.ok(state.arm.base[1] < a3[1] && state.arm.base[1] > a4[1]);
});

test("recorded evidence schedules one simulation and one real pair", () => {
  const simulation = deriveTourState(75_001);
  const robot = deriveTourState(90_001);

  assert.equal(simulation.evidenceCaseId, "pi05-libero10");
  assert.equal(simulation.evidenceLocalTimeMs, 1);
  assert.equal(robot.evidenceCaseId, "corn");
  assert.equal(robot.evidenceLocalTimeMs, 1);
  assert.equal(deriveTourState(TOUR_DURATION_MS).evidenceLocalTimeMs, 15_000);
});

test("Base converges toward the ideal path after a5 but ends just outside the target", () => {
  const { actions } = deriveTourState(24_000);
  for (const action of actions.slice(0, 5)) assert.deepEqual(action.base, action.ideal);
  assert.ok(actions[5].base[1] - actions[5].ideal[1] > 0.25);
  for (let index = 6; index < 8; index += 1) {
    const current = actions[index];
    const previous = actions[index - 1];
    assert.ok(current.base[1] < previous.base[1]);
    assert.ok(current.base[1] - current.ideal[1] < previous.base[1] - previous.ideal[1]);
  }
  const endpointGap = (actions[7].base[1] - actions[7].ideal[1]) * 350;
  assert.ok(endpointGap > 30 && endpointGap < 46, "Base endpoint clears the target edge by a small visible margin");
});

test("refinement corrects local errors without exactly reproducing the ideal trajectory", () => {
  const { actions } = deriveTourState(70_000);
  for (const action of actions.slice(5)) {
    const distance = (point) => Math.hypot(...point.map((value, i) => value - action.ideal[i]));
    assert.ok(distance(action.refined) > 0.01);
    assert.ok(distance(action.refined) < distance(action.base));
  }
  const end = actions[7];
  assert.ok(Math.abs(end.refined[0] - end.ideal[0]) * 760 < 26);
  assert.ok(Math.abs(end.refined[1] - end.ideal[1]) * 350 < 23);
});

test("manual action inspection synchronizes Base, refined, and ideal wrists", () => {
  const state = deriveTourState(67_000, { selectedStep: 5 });
  assert.deepEqual(state.arm.base, state.actions[5].base);
  assert.deepEqual(state.arm.refined, state.actions[5].refined);
  assert.deepEqual(state.arm.ideal, state.actions[5].ideal);
});

test("the ideal arm interpolates on its own trajectory at the same action position", () => {
  const state = deriveTourState(12_500);
  assert.equal(state.actionPosition, 5.5);
  assert.deepEqual(state.arm.ideal, state.actions[5].ideal.map((value, index) =>
    (value + state.actions[6].ideal[index]) / 2));
  assert.notDeepEqual(state.arm.ideal, state.arm.base);
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
