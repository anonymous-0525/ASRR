import test from "node:test";
import assert from "node:assert/strict";

import { deriveTourState } from "../docs/static/js/explainer-model.mjs";
import {
  describeState,
  renderChapterTabs,
  renderScene,
} from "../docs/static/js/explainer-scenes.mjs";
import {
  bindBeforeAfterRelease,
  createTourController,
  mountMediaPair,
} from "../docs/static/js/explainer.js";

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

test("selected action identity is linked across trajectory and token views", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();
  const state = deriveTourState(43_000, { selectedStep: 5 });

  renderScene(stage, inspector, state);

  assert.match(stage.innerHTML, /data-trajectory-id="a5"/);
  assert.match(stage.innerHTML, /data-token-id="a5"/);
  assert.match(inspector.innerHTML, /<code>a5<\/code>/);
});

test("action-only scene omits context while conditioned scene exposes it", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();

  renderScene(stage, inspector, deriveTourState(43_000, { variant: "A" }));
  assert.doesNotMatch(stage.innerHTML, /data-context-path/);
  assert.doesNotMatch(stage.innerHTML, /Policy context/);

  renderScene(stage, inspector, deriveTourState(43_000, { variant: "C" }));
  assert.match(stage.innerHTML, /data-context-path/);
  assert.match(stage.innerHTML, /Policy context/);
});

test("manual variant and before state never mutate authored time", () => {
  const authored = deriveTourState(43_000);
  const manual = { ...authored, variant: "A", before: true, selectedStep: 5 };
  assert.equal(manual.timeMs, authored.timeMs);

  const restored = deriveTourState(manual.timeMs);
  assert.equal(restored.variant, authored.variant);
  assert.equal(restored.before, authored.before);
});

test("chapter tabs and descriptions follow derived state", () => {
  const tabs = new FakeElement();
  const state = deriveTourState(60_000);
  renderChapterTabs(tabs, state);

  assert.equal((tabs.innerHTML.match(/data-chapter-index=/g) ?? []).length, 5);
  assert.match(tabs.innerHTML, /aria-current="step"[^>]*>.*Native Execution/s);
  assert.match(describeState(state), /Native Execution/);
});

test("every chapter names takeaway, input, change, and evidence type", () => {
  const stage = new FakeElement();
  const inspector = new FakeElement();
  for (const timeMs of [1_000, 18_000, 38_000, 58_000, 75_000]) {
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
  controller.seek(99_000);
  assert.equal(controller.getState().timeMs, 90_000);
  assert.equal(controller.getState().playing, false);
  assert.equal(controller.getState().atEnd, true);
});

test("chapter navigation and replay use authored boundaries", () => {
  const { controller } = makeController(43_000);
  controller.nextChapter();
  assert.equal(controller.getState().timeMs, 55_000);
  controller.previousChapter();
  assert.equal(controller.getState().timeMs, 31_000);
  controller.seek(48_000);
  controller.replayChapter();
  assert.equal(controller.getState().timeMs, 31_000);
});

test("manual inspection pauses and play restores authored state", () => {
  const { controller } = makeController(43_000);
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
  assert.equal(restored.scene.variant, deriveTourState(43_000).variant);
  assert.equal(restored.scene.selectedStep, deriveTourState(43_000).selectedStep);
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
  const external = makeClock(72_000);
  controller.setExternalClock(external);
  assert.equal(controller.getState().scene.chapterIndex, 4);
  external.seek(18_000);
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
