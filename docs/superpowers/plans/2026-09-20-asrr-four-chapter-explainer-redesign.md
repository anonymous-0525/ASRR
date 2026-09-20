# ASRR Four-Chapter Explainer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current five-part, 90-second explainer with a four-part, 120-second robot-centered experience and embed the same live experience directly below the project hero.

**Architecture:** Preserve the existing pure-model, scene-renderer, media-controller, and page-controller boundaries. Extend the pure timeline state with sequential action history and arm poses, render all four scenes through reusable SVG primitives, and make the controller initialize either the standalone workbench or a compact homepage shell. Recorded evidence and hero media continue to come from the local evidence manifest.

**Tech Stack:** Static HTML, CSS, browser-native ES modules, SVG, HTML5 video, Node `node:test`, Python `unittest`, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-09-20-asrr-four-chapter-explainer-redesign.md`

## Global Constraints

- The authored timeline is exactly 120,000 ms with four equal 30,000 ms chapters.
- The automatic evidence sequence uses pi0.5 LIBERO-10 and Piper Corn-to-plate paired videos.
- Real-robot evidence retains 3x playback; simulation evidence retains its manifest playback rate.
- Illustrative trajectory differences are clearly labelled and never represented as recorded activations.
- The base policy remains frozen and the native execution rule remains preserved in the teaching copy.
- No framework migration, remote API, model download, browser inference, or new runtime dependency is introduced.
- Homepage and standalone presentations share model, scene, controller, and evidence code.
- Anonymous public assets must not expose local paths, author identities, analytics, or unreported experiments.

## Review Focus

- Exact chapter boundaries, including `30,000`, `60,000`, `90,000`, and `120,000` ms, resolve to the expected chapter without an invalid intermediate state; Task 1 adds boundary tests.
- Arbitrary seeking cannot mark future actions as visited or skip directly from `a0` to a later action at chapter start; Task 1 adds history-state tests.
- Re-rendering a scene while the Base hold control is pressed cannot leave the controller stuck in the Base state; Task 2 preserves the document-level release regression test.
- Browser metadata and scene replacement cannot reset Corn-to-plate videos from 3x or leave stale videos playing; Task 3 adds synchronized-pair tests.
- Multiple explainer roots on the homepage and standalone page cannot share mutable playback or media state; Task 4 adds independent-instance tests.

---

### Task 1: Four-Chapter Timeline And Robot Teaching State

**Files:**
- Modify: `docs/static/js/explainer-model.mjs`
- Modify: `tests/explainer-model.test.mjs`
- Modify: `tests/explainer-controller.test.mjs`

**Interfaces:**
- Produces: `TOUR_DURATION_MS = 120000` and four-entry `CHAPTERS`.
- Produces: `deriveTourState(timeMs, overrides)` with `visitedThrough`, `selectedStep`, `arm.base`, `arm.refined`, `variant`, and four-chapter state.
- Preserves: `composeResidual`, `chapterAtTime`, and `validateTeachingState` exports.

- [ ] **Step 1: Add failing timeline and action-history tests**

```js
test("the redesigned tour has four equal 30-second chapters", () => {
  assert.equal(TOUR_DURATION_MS, 120_000);
  assert.deepEqual(CHAPTERS.map(({ startMs, endMs }) => [startMs, endMs]), [
    [0, 30_000], [30_000, 60_000], [60_000, 90_000], [90_000, 120_000],
  ]);
  assert.equal(chapterAtTime(30_000).id, "editable-interface");
  assert.equal(chapterAtTime(60_000).id, "residual-refinement");
  assert.equal(chapterAtTime(90_000).id, "recorded-evidence");
  assert.equal(chapterAtTime(120_000).id, "recorded-evidence");
});

test("local error begins at a3 and walks through a7 without skipping", () => {
  assert.equal(deriveTourState(0).selectedActionId, "a3");
  assert.equal(deriveTourState(29_999).selectedActionId, "a7");
  for (let index = 3; index < 8; index += 1) {
    const state = deriveTourState((index - 3) * 6_000 + 1);
    assert.equal(state.selectedStep, index);
    assert.equal(state.visitedThrough, index - 1);
  }
});
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs`

Expected: failures report the old `90_000` duration, five chapters, and missing `visitedThrough`.

- [ ] **Step 3: Implement four-chapter authored state**

Replace `CHAPTERS` with `local-error`, `editable-interface`, `residual-refinement`, and `recorded-evidence`, each lasting 30 seconds. Compute the local-error action as:

```js
function localErrorStep(progress) {
  return Math.min(7, Math.floor(clamp(progress, 0, 0.999999) * 8));
}

const selectedStep = chapter.index === 0
  ? localErrorStep(chapterProgress)
  : authoredStep(chapter.index, chapterProgress);
const visitedThrough = chapter.index === 0 ? selectedStep - 1 : selectedStep;
```

Add deterministic two-link arm inputs for base and refined end-effector points while keeping geometry calculation in the scene module:

```js
arm: {
  base: [...actions[selectedStep].base],
  refined: [...actions[selectedStep].refined],
},
```

Make ASRR-A active from 60-75 seconds and ASRR-C active from 75-90 seconds. Keep manual overrides unchanged.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs`

Expected: all model and controller tests pass with the new boundaries.

- [ ] **Step 5: Commit the timeline state**

```bash
git add docs/static/js/explainer-model.mjs tests/explainer-model.test.mjs tests/explainer-controller.test.mjs
git commit -m "Redesign explainer timeline state"
```

### Task 2: Robot-Centered Four-Scene Rendering

**Files:**
- Modify: `docs/static/js/explainer-scenes.mjs`
- Modify: `docs/static/css/explainer.css`
- Modify: `tests/explainer-model.test.mjs`
- Modify: `tests/explainer-controller.test.mjs`

**Interfaces:**
- Consumes: four-chapter state and `state.arm` from Task 1.
- Produces: `armGeometry(target, options)` returning base, elbow, wrist, and gripper SVG coordinates.
- Produces: four scene renderers selected by `state.chapterIndex`.
- Preserves: `renderScene`, `renderChapterTabs`, and `describeState` exports.

- [ ] **Step 1: Add failing tests for scene semantics**

Add assertions that rendered local-error markup contains all eight action labels, one `is-current`, completed actions through `visitedThrough`, and an SVG arm. Add assertions that editable-interface markup contains both `Adapt the policy` and `Edit the proposal`, contains `Editable action proposal`, and does not contain `token-strip`. Add assertions that residual markup contains a ghost Base arm, solid Refined arm, and only exposes policy context for ASRR-C.

```js
assert.match(localMarkup, /data-arm="active"/);
assert.equal((localMarkup.match(/data-action-label=/g) ?? []).length, 8);
assert.match(interfaceMarkup, /Adapt the policy/);
assert.match(interfaceMarkup, /Edit the proposal/);
assert.doesNotMatch(interfaceMarkup, /token-strip/);
assert.match(contextMarkup, /Policy-side data/);
assert.doesNotMatch(actionOnlyMarkup, /Policy-side data/);
```

- [ ] **Step 2: Run tests and confirm semantic failures**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs`

Expected: failures report missing arm and comparison markup.

- [ ] **Step 3: Add reusable arm and action-history SVG helpers**

Implement planar arm geometry with fixed link lengths and clamped reach:

```js
export function armGeometry([x, y], {
  origin = [155, 386], upper = 112, lower = 98,
} = {}) {
  const dx = x - origin[0];
  const dy = y - origin[1];
  const reach = Math.min(upper + lower - 1, Math.max(24, Math.hypot(dx, dy)));
  const heading = Math.atan2(dy, dx);
  const elbowOffset = Math.acos((upper ** 2 + reach ** 2 - lower ** 2) / (2 * upper * reach));
  const elbow = [
    origin[0] + upper * Math.cos(heading + elbowOffset),
    origin[1] + upper * Math.sin(heading + elbowOffset),
  ];
  return { base: origin, elbow, wrist: [x, y] };
}
```

Render the current point in dark gold, visited points in pale gold, future points in neutral gold, with text labels `a0` through `a7` always visible.

- [ ] **Step 4: Replace Editable Proposal with the split adaptation interface**

Render two un-nested full-height lanes. The left lane animates observation/instruction through an updateable policy into direct robot action. The right lane marks the policy as frozen, labels its output `Editable action proposal`, passes it through the compact refiner, and drives a corrected arm. Do not render action tokens in this chapter.

- [ ] **Step 5: Replace Residual Refinement with overlaid arm comparison**

Render a low-opacity Base arm and gold trajectory underneath an opaque Refined arm and teal trajectory. Increase the illustrative residual displacement for the local-error teaching geometry only, add coral curved residual arrows, and mark the view `Illustrative, displacement enlarged for clarity`. For ASRR-C, add one policy-data block and animated context-to-refiner lines. End with `Base policy frozen` and `Native execution preserved` status badges.

- [ ] **Step 6: Style four scenes responsively**

Add stable stage dimensions, gold history tokens, arm joints and links, split interface lanes, ghost/refined arm roles, curved residual arrows, and compact status badges. Keep desktop content within the existing stage height and use an internal horizontal viewport for narrow mobile comparisons. Add reduced-motion rules that remove interpolation while retaining state transitions.

- [ ] **Step 7: Run focused tests and verify rendering behavior**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs`

Expected: all tests pass, including the existing document-level Base-hold release test.

- [ ] **Step 8: Commit the redesigned scenes**

```bash
git add docs/static/js/explainer-scenes.mjs docs/static/css/explainer.css tests/explainer-model.test.mjs tests/explainer-controller.test.mjs
git commit -m "Render robot-centered ASRR explainer scenes"
```

### Task 3: Authored Paired-Video Evidence

**Files:**
- Modify: `docs/static/data/explainer-evidence.json`
- Modify: `docs/static/js/explainer-media.mjs`
- Modify: `docs/static/js/explainer-scenes.mjs`
- Modify: `docs/static/js/explainer.js`
- Modify: `tests/explainer-media.test.mjs`
- Modify: `tests/test_static_site.py`

**Interfaces:**
- Consumes: `pi05-libero10` and `corn` evidence cases.
- Produces: evidence chapter phases `simulation` for 90-105 seconds and `real_robot` for 105-120 seconds.
- Produces: media controller `loadPair(caseData)`, `sync(timeSeconds, playing)`, and `clear()` behavior for authored scene media.

- [ ] **Step 1: Add failing tests for evidence scheduling and synchronization**

Add tests confirming the first evidence half selects `pi05-libero10`, the second selects `corn`, metadata events preserve manifest playback rates, stale pairs stop after a phase change, and the real pair stays at 3x.

```js
test("recorded evidence schedules one simulation and one real pair", () => {
  assert.equal(deriveTourState(90_001).evidenceCaseId, "pi05-libero10");
  assert.equal(deriveTourState(105_001).evidenceCaseId, "corn");
});
```

- [ ] **Step 2: Run media and model tests and confirm failure**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-media.test.mjs`

Expected: missing `evidenceCaseId` and authored evidence synchronization fail.

- [ ] **Step 3: Add authored evidence fields and manifest labels**

Add `evidenceCaseId` and `evidenceLocalTimeMs` to derived state. Keep all five existing cases in the manifest, with pi0.5 and Corn flagged `authored: true`. Preserve Corn playback rate `3` and pi0.5 playback rate `1`.

- [ ] **Step 4: Mount paired videos directly in the evidence stage**

Reuse `mountMediaPair` for inline stage media. On case changes, clear stale nodes before loading the next pair. Drive both current times from `evidenceLocalTimeMs`, scaled by manifest playback rate, and maintain the existing retry/poster fallback. Render measured metric chips over the media without covering manipulated objects or terminal outcomes.

- [ ] **Step 5: Keep secondary evidence accessible**

Add a `More evidence` command that opens the existing modal wall for OpenVLA-OFT, Block-to-holder, and Block stacking. The automatic 120-second tour must not open this modal.

- [ ] **Step 6: Run media and static-site tests**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-media.test.mjs && python -m unittest tests.test_static_site -v`

Expected: synchronized-pair, 3x, retry, anonymity, and asset-scope tests pass.

- [ ] **Step 7: Commit paired evidence**

```bash
git add docs/static/data/explainer-evidence.json docs/static/js/explainer-media.mjs docs/static/js/explainer-scenes.mjs docs/static/js/explainer.js tests/explainer-media.test.mjs tests/test_static_site.py
git commit -m "Show paired evidence in the authored tour"
```

### Task 4: Reusable Standalone And Embedded Explainer Shells

**Files:**
- Modify: `docs/explainer.html`
- Modify: `docs/static/js/explainer.js`
- Modify: `docs/static/css/explainer.css`
- Modify: `tests/explainer-controller.test.mjs`
- Modify: `tests/test_static_site.py`

**Interfaces:**
- Produces: `mountExplainer(root, options)` returning `{ controller, destroy }`.
- Consumes: `options.mode` as `"standalone" | "embedded"` and `options.autoplayWhenVisible`.
- Preserves: `createTourController` as the pure controller interface.

- [ ] **Step 1: Add failing independent-instance tests**

Create two controllers with separate fake clocks, seek one, and assert the other remains at zero. Add static tests requiring a `[data-explainer-root]` standalone shell and an embedded shell on the homepage without an iframe.

- [ ] **Step 2: Run controller and static tests and confirm failure**

Run: `node --test tests/explainer-controller.test.mjs && python -m unittest tests.test_static_site -v`

Expected: `mountExplainer` and embedded-root assertions fail.

- [ ] **Step 3: Refactor page initialization into `mountExplainer`**

Move document queries below a root parameter and replace global `getElementById` calls with root-scoped selectors. Instantiate media, event handlers, visibility handlers, and IntersectionObserver per root. `destroy()` removes listeners, disconnects observers, pauses the clock, and clears media.

- [ ] **Step 4: Update standalone shell for four chapters and two minutes**

Replace five hard-coded tabs with an accessible four-button chapter container populated by `renderChapterTabs`, update metadata to `two-minute interactive explanation`, and retain complete playback and inspector controls.

- [ ] **Step 5: Add embedded-mode behavior**

Embedded mode uses the same stage, inspector, timeline, and chapter navigation with a compact header. It begins muted playback only after at least 55% of the root is visible, pauses below 20% visibility, and never restarts after a user manually pauses. Include an `Open full experience` link.

- [ ] **Step 6: Run focused tests**

Run: `node --test tests/explainer-controller.test.mjs && python -m unittest tests.test_static_site -v`

Expected: independent controller, root scoping, and embedded shell tests pass.

- [ ] **Step 7: Commit reusable shells**

```bash
git add docs/explainer.html docs/static/js/explainer.js docs/static/css/explainer.css tests/explainer-controller.test.mjs tests/test_static_site.py
git commit -m "Embed the shared ASRR explainer runtime"
```

### Task 5: Homepage Live Explainer And Moving Hero Media

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/static/css/site.css`
- Modify: `docs/static/js/site.js`
- Modify: `tests/test_static_site.py`

**Interfaces:**
- Consumes: `mountExplainer` from Task 4 and existing local evidence media.
- Produces: one embedded explainer immediately below the hero and four labelled hero video tiles.
- Preserves: current navigation anchors, paper/results content, and standalone explainer link.

- [ ] **Step 1: Add failing homepage structure tests**

Require four hero videos with `muted`, `loop`, `playsinline`, and local poster fallbacks. Require the embedded explainer immediately after the hero and assert the old `.explainer-preview` section is absent.

```python
self.assertNotIn('class="explainer-preview"', html)
self.assertEqual(html.count('class="hero__tile-video"'), 4)
self.assertLess(html.index('data-explainer-root="embedded"'), html.index('id="abstract"'))
```

- [ ] **Step 2: Run static-site tests and confirm failure**

Run: `python -m unittest tests.test_static_site -v`

Expected: old preview and static hero markup violate new assertions.

- [ ] **Step 3: Replace the hero mosaic with paired looping videos**

Use local pi0.5 Base/ASRR and Corn Base/ASRR clips. Each tile contains a poster, a fixed Base or ASRR label, and semantic task label. `site.js` starts tiles when the hero is visible, pauses them when hidden, reapplies playback rate after metadata, and leaves posters static under `prefers-reduced-motion: reduce`.

- [ ] **Step 4: Move the live explainer directly below the hero**

Remove the current static preview section. Add the compact embedded shell before Abstract, import `mountExplainer`, and initialize it with:

```js
mountExplainer(root, {
  mode: "embedded",
  autoplayWhenVisible: true,
});
```

- [ ] **Step 5: Style the moving hero and embedded workbench**

Keep the hero media on the right at desktop widths and below the title on mobile. Use stable 4:3 tile ratios, no text overlap, poster fallback, and no page-wide overflow. Give the embedded explainer a full-width band rather than a decorative nested card.

- [ ] **Step 6: Run static tests and local-reference checks**

Run: `python -m unittest tests.test_static_site -v`

Expected: homepage placement, local media, anonymous content, and reduced-motion hooks pass.

- [ ] **Step 7: Commit homepage integration**

```bash
git add docs/index.html docs/static/css/site.css docs/static/js/site.js tests/test_static_site.py
git commit -m "Place the live ASRR tour on the homepage"
```

### Task 6: Full Verification, Visual Acceptance, And Publication

**Files:**
- Modify if required by verified defects: files touched in Tasks 1-5
- Modify: `docs/README.md`
- Test: `tests/explainer-model.test.mjs`
- Test: `tests/explainer-controller.test.mjs`
- Test: `tests/explainer-media.test.mjs`
- Test: `tests/test_static_site.py`

**Interfaces:**
- Verifies: four chapters, embedded/standalone parity, paired evidence, hero media, responsive layout, and anonymous deployment.

- [ ] **Step 1: Run the complete automated suite**

```bash
node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs tests/explainer-media.test.mjs
python -m unittest discover -s tests -v
python -m compileall -q asrr_core examples tests
git diff --check
```

Expected: all available tests pass; PyTorch-dependent tests may skip only when PyTorch is unavailable.

- [ ] **Step 2: Start the local static server**

Run: `python -m http.server 8765 --directory docs`

Open: `http://127.0.0.1:8765/` and `http://127.0.0.1:8765/explainer.html`.

- [ ] **Step 3: Verify complete authored playback**

At 1x, confirm chapter transitions at 30, 60, 90, and 120 seconds; Local Error begins at `a0`; ASRR-A changes to ASRR-C at 75 seconds; pi0.5 changes to Corn at 105 seconds; Replay returns to `a0`.

- [ ] **Step 4: Verify nonlinear interactions**

Seek to every boundary, switch ASRR-A/C, select `a7`, press and release Base outside the redrawn control, pause/resume, switch tabs, force a video failure and retry, and open/close More evidence. Confirm no stale arm, control, or video state persists.

- [ ] **Step 5: Verify desktop and mobile visual layout**

Inspect dark and light standalone views at 1440x900 and homepage/standalone views at 390x844. Confirm no incoherent overlap, desktop stage overflow, clipped action labels, unreadable arm geometry, or page-wide mobile overflow. Confirm all text fits controls.

- [ ] **Step 6: Verify reduced motion and autoplay constraints**

With reduced motion enabled, confirm hero posters remain static and chapter state still changes. Block autoplay and confirm explicit Play starts both the tour and evidence media. Hide the tab and confirm clocks and all videos pause.

- [ ] **Step 7: Update project documentation**

Update `docs/README.md` from a 90-second five-chapter tour to a 120-second four-chapter tour, name the homepage embedding, and document the authored pi0.5/Corn evidence sequence.

- [ ] **Step 8: Commit verified release**

```bash
git add docs tests package.json
git commit -m "Finalize four-chapter ASRR explainer"
```

- [ ] **Step 9: Push and verify GitHub Pages**

Push `main` using the anonymous account, wait for Pages deployment, then verify the homepage, standalone explainer, JavaScript modules, JSON manifest, PNG posters, and MP4 range requests return successful responses with correct MIME types.
