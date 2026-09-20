import test from "node:test";
import assert from "node:assert/strict";

import { deriveTourState } from "../docs/static/js/explainer-model.mjs";
import {
  describeState,
  renderChapterTabs,
  renderScene,
} from "../docs/static/js/explainer-scenes.mjs";

class FakeElement {
  constructor() {
    this.innerHTML = "";
    this.textContent = "";
  }
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
