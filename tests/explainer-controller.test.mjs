import test from "node:test";
import assert from "node:assert/strict";

import { CHAPTERS, TOUR_DURATION_MS, deriveTourState } from "../docs/static/js/explainer-model.mjs";
import * as scenes from "../docs/static/js/explainer-scenes.mjs";
const { describeState, renderChapterTabs, renderScene } = scenes;
import * as explainerModule from "../docs/static/js/explainer.js";
const {
  bindBeforeAfterRelease,
  createTourController,
  mountMediaPair,
} = explainerModule;

class FakeElement {
  constructor() {
    this.innerHTML = "";
    this.textContent = "";
  }
}

function makeClock(initial = 0) {
  let timeMs = initial;
  let playing = false;
  const listeners = new Set();
  const emit = () => listeners.forEach((listener) => listener(timeMs, playing));
  return {
    getTimeMs: () => timeMs,
    isPlaying: () => playing,
    play() { playing = true; emit(); },
    pause() { playing = false; emit(); },
    seek(next) { timeMs = next; emit(); },
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    tick(next) { timeMs = next; emit(); },
  };
}

function makeController(initial = 0, storedTheme = null) {
  const clock = makeClock(initial);
  const renders = [];
  const stored = new Map(storedTheme ? [["asrr-explainer-theme", storedTheme]] : []);
  const storage = {
    getItem: (key) => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, value),
  };
  const media = { sync() {}, pause() {} };
  const controller = createTourController({
    model: { deriveTourState },
    renderer: (state) => renders.push(state),
    media,
    storage,
    clock,
  });
  return { controller, clock, renders, stored };
}

test("local error renders every labelled action and begins the active sequence at a3", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();
  const state = deriveTourState(0);

  renderScene(stage, inspector, state);

  assert.equal(state.selectedActionId, "a3");
  assert.match(stage.innerHTML, /data-arm="active"/);
  assert.match(stage.innerHTML, /data-arm="ideal-ghost"/);
  assert.match(stage.innerHTML, /data-path="ideal"/);
  assert.match(stage.innerHTML, /Local error at a5/);
  assert.match(stage.innerHTML, /missed target/);
  assert.equal((stage.innerHTML.match(/data-action-label=/g) ?? []).length, 8);
  assert.equal((stage.innerHTML.match(/is-visited/g) ?? []).length, state.visitedThrough + 1);
  assert.equal((stage.innerHTML.match(/is-current/g) ?? []).length, 1);
  assert.match(stage.innerHTML, new RegExp(`data-trajectory-id="${state.selectedActionId}"`));
  assert.match(inspector.innerHTML, new RegExp(`<code>${state.selectedActionId}<\\/code>`));
});

test("arm geometry reaches the requested wrist with stable joints", () => {
  assert.equal(typeof scenes.armGeometry, "function");
  const geometry = scenes.armGeometry([340, 250]);
  assert.deepEqual(geometry.base, [345, 405]);
  assert.deepEqual(geometry.wrist, [340, 250]);
  assert.equal(geometry.elbow.length, 2);
  assert.ok(geometry.elbow.every(Number.isFinite));
  for (let t = 0; t < 75_000; t += 250) {
    const state = deriveTourState(t);
    for (const position of Object.values(state.arm)) {
      const target = [70 + position[0] * 760, 54 + position[1] * 350];
      const arm = scenes.armGeometry(target);
      const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
      assert.ok(Math.abs(distance(arm.base, arm.elbow) - 340) < 1e-6);
      assert.ok(Math.abs(distance(arm.elbow, arm.wrist) - 220) < 1e-6);
      assert.ok(arm.elbow[0] > 35 && arm.elbow[0] < 865);
      assert.ok(arm.elbow[1] > 35 && arm.elbow[1] < 455);
    }
  }
});

test("editable interface contrasts policy updates with proposal editing", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();

  renderScene(stage, inspector, deriveTourState(45_000));

  assert.match(stage.innerHTML, /Adapt the policy/);
  assert.match(stage.innerHTML, /Edit the proposal/);
  assert.match(stage.innerHTML, /Editable action proposal/);
  assert.equal((stage.innerHTML.match(/data-action-label=/g) ?? []).length, 16);
  assert.match(stage.innerHTML, /trajectory--refined/);
  assert.equal((stage.innerHTML.match(/data-arm="ideal-ghost"/g) ?? []).length, 2);
  assert.equal((stage.innerHTML.match(/data-path="ideal"/g) ?? []).length, 2);
  assert.doesNotMatch(stage.innerHTML, /token-strip/);
});

test("residual scene overlays both arms and exposes policy data only for ASRR-C", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();

  renderScene(stage, inspector, deriveTourState(70_000, { variant: "A" }));
  assert.match(stage.innerHTML, /residual-mode-row--A/);
  assert.match(stage.innerHTML, /data-arm="base-ghost"/);
  assert.match(stage.innerHTML, /data-arm="refined"/);
  assert.match(stage.innerHTML, /data-path="ideal"/);
  assert.doesNotMatch(stage.innerHTML, /data-arm="ideal-ghost"/);
  assert.match(stage.innerHTML, /residual-curve/);
  assert.doesNotMatch(stage.innerHTML, /data-context-path/);
  assert.doesNotMatch(stage.innerHTML, /Policy-side data/);
  assert.doesNotMatch(stage.innerHTML, /data-policy-state-at/);

  renderScene(stage, inspector, deriveTourState(70_000, { variant: "C" }));
  assert.match(stage.innerHTML, /residual-mode-row--C/);
  assert.match(stage.innerHTML, /class="context-path"/);
  assert.match(stage.innerHTML, /data-context-path/);
  assert.match(stage.innerHTML, /Policy-side data/);
  assert.match(stage.innerHTML, /Native execution preserved/);
  assert.match(stage.innerHTML, /data-policy-state-at="a5"/);
  assert.match(stage.innerHTML, /Policy state used at a5/);
});

test("recorded evidence links directly to homepage videos and results in both modes", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();
  for (const time of [79_000, 94_000, TOUR_DURATION_MS]) {
    renderScene(stage, inspector, deriveTourState(time));
    assert.match(stage.innerHTML, /href="\.\/#videos"/);
    assert.match(stage.innerHTML, /href="\.\/#results"/);
    assert.doesNotMatch(stage.innerHTML, /More evidence|data-open-evidence-library/);
    renderScene(stage, inspector, deriveTourState(time), { homePath: "" });
    assert.match(stage.innerHTML, /href="#videos"/);
    assert.match(stage.innerHTML, /href="#results"/);
  }
});

test("manual variant and before state never mutate authored time", () => {
  const authored = deriveTourState(70_000);
  const manual = { ...authored, variant: "A", before: true, selectedStep: 5 };
  assert.equal(manual.timeMs, authored.timeMs);

  const restored = deriveTourState(manual.timeMs);
  assert.equal(restored.variant, authored.variant);
  assert.equal(restored.before, authored.before);
});

test("separate explainer controllers keep independent clocks and scene state", () => {
  const first = makeController();
  const second = makeController();

  first.controller.seek(72_000);

  assert.equal(first.controller.getState().timeMs, 72_000);
  assert.equal(second.controller.getState().timeMs, 0);
  assert.equal(typeof explainerModule.mountExplainer, "function");
});

test("chapter tabs and descriptions follow derived state", () => {
  const tabs = new FakeElement();
  const state = deriveTourState(60_000);
  renderChapterTabs(tabs, state);

  assert.equal((tabs.innerHTML.match(/data-chapter-index=/g) ?? []).length, 4);
  assert.match(tabs.innerHTML, /aria-current="step"[^>]*>.*Residual Refinement/s);
  assert.match(describeState(state), /Residual Refinement/);
});

test("every chapter names takeaway, input, change, and evidence type", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();
  for (const timeMs of [1_000, 31_000, 61_000, 91_000]) {
    renderScene(stage, inspector, deriveTourState(timeMs));
    assert.match(inspector.innerHTML, /Takeaway/);
    assert.match(inspector.innerHTML, /Input/);
    assert.match(inspector.innerHTML, /Change/);
    assert.match(inspector.innerHTML, /Evidence/);
  }
});

test("play, pause, seek, and end state follow the shared clock", () => {
  const { controller, clock } = makeController();
  controller.play();
  assert.equal(controller.getState().playing, true);
  clock.tick(45_000);
  assert.equal(controller.getState().timeMs, 45_000);
  controller.pause();
  assert.equal(controller.getState().playing, false);
  controller.seek(12_000);
  assert.equal(controller.getState().timeMs, 12_000);
  controller.seek(129_000);
  assert.equal(controller.getState().timeMs, TOUR_DURATION_MS);
  assert.equal(controller.getState().playing, false);
  assert.equal(controller.getState().atEnd, true);
});

test("chapter navigation and replay use authored boundaries", () => {
  const { controller } = makeController(70_000);
  controller.nextChapter();
  assert.equal(controller.getState().timeMs, CHAPTERS[3].startMs);
  controller.previousChapter();
  assert.equal(controller.getState().timeMs, CHAPTERS[2].startMs);
  controller.seek(68_000);
  controller.replayChapter();
  assert.equal(controller.getState().timeMs, CHAPTERS[2].startMs);
});

test("manual inspection pauses and play restores authored state", () => {
  const { controller } = makeController(70_000);
  controller.play();
  controller.setVariant("A");
  controller.selectStep(7);
  controller.setBefore(true);

  const manual = controller.getState();
  assert.equal(manual.playing, false);
  assert.equal(manual.scene.variant, "A");
  assert.equal(manual.scene.selectedStep, 7);
  assert.equal(manual.scene.before, true);

  controller.play();
  const restored = controller.getState();
  assert.equal(restored.scene.variant, deriveTourState(70_000).variant);
  assert.equal(restored.scene.selectedStep, deriveTourState(70_000).selectedStep);
  assert.equal(restored.scene.before, false);
});

test("hidden tab and Space outside a form control pause or resume", () => {
  const { controller } = makeController();
  controller.play();
  controller.onVisibilityChange(true);
  assert.equal(controller.getState().playing, false);

  let prevented = false;
  controller.handleKey({ code: "Space", target: { tagName: "DIV" }, preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(controller.getState().playing, true);

  controller.handleKey({ code: "Space", target: { tagName: "INPUT" }, preventDefault() { throw new Error("should not run"); } });
  assert.equal(controller.getState().playing, true);
});

test("invalid stored theme falls back to dark and explicit changes persist", () => {
  const { controller, stored } = makeController(0, "sepia");
  assert.equal(controller.getState().theme, "dark");
  controller.setTheme("light");
  assert.equal(controller.getState().theme, "light");
  assert.equal(stored.get("asrr-explainer-theme"), "light");
});

test("an external clock adapter drives the same scene state", () => {
  const { controller } = makeController();
  const external = makeClock(105_000);
  controller.setExternalClock(external);
  assert.equal(controller.getState().scene.chapterIndex, 3);
  external.seek(32_000);
  assert.equal(controller.getState().scene.chapterIndex, 1);
});

test("retry remount replaces stale visible media nodes", () => {
  const container = {
    children: ["stale"],
    replaceChildren() { this.children = []; },
    append(child) { this.children.push(child); },
  };
  const documentRef = {
    createElement(tagName) {
      return {
        tagName,
        children: [],
        textContent: "",
        append(...children) { this.children.push(...children); },
      };
    },
  };
  const videos = [{ id: "new-base" }, { id: "new-asrr" }];

  mountMediaPair(container, videos, documentRef);

  assert.equal(container.children.length, 2);
  assert.equal(container.children[0].children[0], videos[0]);
  assert.equal(container.children[1].children[0], videos[1]);
});

test("document-level pointer release clears a redrawn Before control", () => {
  const listeners = new Map();
  const documentRef = {
    addEventListener(type, listener) { listeners.set(type, listener); },
  };
  let before = true;
  const controller = {
    getState: () => ({ scene: { before } }),
    setBefore(value) { before = value; },
  };
  bindBeforeAfterRelease(documentRef, controller);

  listeners.get("pointerup")({});

  assert.equal(before, false);
});
