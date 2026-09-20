# ASRR Interactive Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 90-second, five-chapter ASRR interactive explainer and a compact homepage entry while preserving the existing anonymous static project page.

**Architecture:** Keep GitHub Pages and the current framework-free site. Pure ES modules derive ASRR teaching values and all tour state from one seekable clock; focused render and media modules consume that state. The homepage links to a dedicated full-workbench `explainer.html`, and recorded evidence remains separate from deterministic illustrative computation.

**Tech Stack:** Static HTML5, CSS custom properties, SVG/DOM, browser ES modules, Node.js built-in test runner, Python `unittest`, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-09-20-asrr-interactive-explainer-design.md`

## Global Constraints

- Keep the site anonymous and publish only through `anonymous-0525/ASRR` on `main` when publication is authorized.
- Use the existing static GitHub Pages stack; add no framework, browser inference, model weights, API key, or runtime service.
- The authored silent tour is 90,000 ms and has exactly five chapters: Local Error, Editable Proposal, Residual Refinement, Native Execution, and Evidence.
- Dark is the initial explainer theme; light mode is available and stored under `asrr-explainer-theme`.
- Preserve paper color semantics: Base blue, ASRR teal, residual coral, context lavender, neutral gray.
- Label deterministic values and trajectory geometry as illustrative; label recorded clips and measured metrics as evidence.
- Use the verified headline values `+14.4 pp`, `62.0 -> 80.9`, `25-63% less recorded compute`, and `+6.7 pp` without adding stronger claims.
- Do not expose narration controls in this phase; provide a clock adapter that later accepts audio time.
- Respect reduced motion, keyboard navigation, mobile layout, hidden-tab pause, and GitHub Pages deployment under `/ASRR/`.
- Preserve the existing paper, results, videos, code, and citation sections except for the new explainer entry.

## Review Focus

- Seeking to `0`, `90,000`, or an exact chapter boundary must restore a valid chapter and never produce negative or out-of-range scene progress; Task 1 pins these boundaries.
- Manual chapter, timestep, ASRR-A/C, and Before/After interaction must pause the guide and resume from authored state rather than leaking manual state; Task 4 tests restoration.
- A media load rejected after the user seeks or exits must not restart stale playback, and a current failure must expose retry while schematic playback continues; Task 5 tests cancellation and retry.
- Missing or malformed evidence entries must fail validation before rendering instead of creating broken or misleading cards; Task 2 tests manifest validation.
- At 390 px and with reduced motion, the page must remain readable and preserve meaningful state changes without page-wide horizontal overflow; Task 7 verifies both conditions.

---

## File Structure

### New files

- `docs/explainer.html` — accessible workbench shell and controls.
- `docs/static/css/explainer.css` — theme tokens, workbench layout, responsive rules, focus, and reduced motion.
- `docs/static/js/explainer-model.mjs` — pure teaching data, ASRR composition, chapter timing, and derived scene state.
- `docs/static/js/explainer-scenes.mjs` — SVG/DOM renderers for the five chapters.
- `docs/static/js/explainer-media.mjs` — lazy media loading, clock synchronization, cancellation, retry, and cleanup.
- `docs/static/js/explainer.js` — DOM bindings, shared clock, playback controls, manual inspection, and theme persistence.
- `docs/static/data/explainer-evidence.json` — measured metrics, clip identities, labels, playback rates, and evidence notes.
- `docs/static/images/explainer/` — posters derived from already authorized project media.
- `tests/explainer-model.test.mjs` — pure scientific and timeline tests.
- `tests/explainer-controller.test.mjs` — controller restoration and interaction tests.
- `tests/explainer-media.test.mjs` — media cancellation, retry, and synchronization tests.
- `docs/THIRD_PARTY.md` — references and any adapted-code attribution.

### Modified files

- `docs/index.html` — explainer preview and primary entry between Abstract and Method.
- `docs/static/css/site.css` — responsive homepage preview styles.
- `tests/test_static_site.py` — resource, anonymity, manifest, entry-link, and viewport-scope checks.
- `docs/README.md` — local explainer preview and validation commands.

## Task 1: Pure ASRR Teaching Model And Tour Timeline

**Files:**
- Create: `docs/static/js/explainer-model.mjs`
- Create: `tests/explainer-model.test.mjs`

**Interfaces:**
- Produces: `TOUR_DURATION_MS`, `CHAPTERS`, `TEACHING_SEQUENCE`, `composeResidual(base, delta, alpha, mask, bound)`, `chapterAtTime(timeMs)`, `deriveTourState(timeMs, overrides = {})`, and `validateTeachingState(state)`.
- Consumes: no browser APIs and no DOM; later tasks import these pure exports.

- [ ] **Step 1: Write failing tests for composition, mask, bound, and boundaries**

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  TOUR_DURATION_MS,
  CHAPTERS,
  composeResidual,
  chapterAtTime,
  deriveTourState,
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
```

- [ ] **Step 2: Run the focused test and confirm missing-module failure**

Run: `node --test tests/explainer-model.test.mjs`

Expected: FAIL because `explainer-model.mjs` does not exist.

- [ ] **Step 3: Implement deterministic teaching data and pure timeline state**

Implement:

```js
export const TOUR_DURATION_MS = 90_000;
export const CHAPTERS = Object.freeze([
  { index: 0, id: "local-error", label: "Local Error", startMs: 0, endMs: 16_000 },
  { index: 1, id: "editable-proposal", label: "Editable Proposal", startMs: 16_000, endMs: 31_000 },
  { index: 2, id: "residual-refinement", label: "Residual Refinement", startMs: 31_000, endMs: 55_000 },
  { index: 3, id: "native-execution", label: "Native Execution", startMs: 55_000, endMs: 70_000 },
  { index: 4, id: "evidence", label: "Evidence", startMs: 70_000, endMs: 90_000 },
]);
```

Use an eight-timestep illustrative 2D trajectory with stable IDs `a0` through `a7`, one local-error marker, per-step residuals, mask values, and optional context values. Clamp `timeMs` into `[0, TOUR_DURATION_MS]`; at the final endpoint resolve the Evidence chapter with progress `1`. `composeResidual` clamps each delta coordinate before multiplying by `alpha` and `mask`, matching the explainer's declared unscaled bound.

- [ ] **Step 4: Extend tests for arbitrary seek and state invariants**

Add tests that seek in descending and non-monotonic order, check stable action IDs, verify ASRR-A removes context terms, verify ASRR-C includes them, and call `validateTeachingState` for 100 evenly spaced timestamps.

- [ ] **Step 5: Run tests**

Run: `node --test tests/explainer-model.test.mjs`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/static/js/explainer-model.mjs tests/explainer-model.test.mjs
git commit -m "Add ASRR explainer teaching model"
```

## Task 2: Evidence Manifest And Static Validation

**Files:**
- Create: `docs/static/data/explainer-evidence.json`
- Create: `docs/static/images/explainer/pi05-base.png`
- Create: `docs/static/images/explainer/pi05-asrr.png`
- Create: `docs/static/images/explainer/openvla-base.png`
- Create: `docs/static/images/explainer/openvla-asrr.png`
- Create: `docs/static/images/explainer/corn-base.png`
- Create: `docs/static/images/explainer/corn-asrr.png`
- Create: `docs/static/images/explainer/holder-base.png`
- Create: `docs/static/images/explainer/holder-asrr.png`
- Create: `docs/static/images/explainer/stack-base.png`
- Create: `docs/static/images/explainer/stack-asrr.png`
- Modify: `tests/test_static_site.py`

**Interfaces:**
- Produces: JSON object `{ metrics, cases }`; each case has `id`, `policy`, `task`, `kind`, `baseVideo`, `asrrVideo`, `basePoster`, `asrrPoster`, `playbackRate`, and `note`.
- Consumes: only existing approved videos under `docs/static/videos/`; no new experiment output.

- [ ] **Step 1: Add a failing manifest contract test**

```python
def test_explainer_evidence_manifest_is_complete(self):
    payload = json.loads((DOCS / "static/data/explainer-evidence.json").read_text())
    self.assertEqual(
        {"pi05-libero10", "openvla-goal", "corn", "holder", "stack"},
        {case["id"] for case in payload["cases"]},
    )
    for case in payload["cases"]:
        self.assertIn(case["kind"], {"simulation", "real_robot"})
        self.assertGreater(case["playbackRate"], 0)
        for key in ("baseVideo", "asrrVideo", "basePoster", "asrrPoster"):
            self.assertTrue((DOCS / case[key]).is_file(), (case["id"], key))
```

Also assert the exact four headline metrics and reject absolute paths, unknown keys, duplicate IDs, empty evidence notes, and any `Octo` or `SmolVLA` label.

- [ ] **Step 2: Run the test and confirm missing-manifest failure**

Run: `python -m unittest discover -s tests -p 'test_static_site.py' -v`

Expected: FAIL because the manifest does not exist.

- [ ] **Step 3: Create traceable posters and manifest**

Derive each poster from an informative frame of its existing source clip without changing the outcome or hiding the robot, object, goal, or terminal state. Use 16:9 or 4:3 images according to the original camera; do not force one crop across tasks. Record the existing clip paths and rates (`1` for simulation clips and `3` for the six real-robot clips).

The metrics object must be:

```json
{
  "pi05_mean_gain_pp": 14.4,
  "pi05_5k_success": [62.0, 80.9],
  "recorded_compute_reduction_percent": [25, 63],
  "real_robot_gain_pp": 6.7
}
```

- [ ] **Step 4: Run static tests**

Run: `python -m unittest discover -s tests -p 'test_static_site.py' -v`

Expected: all static-site tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/static/data/explainer-evidence.json docs/static/images/explainer tests/test_static_site.py
git commit -m "Add traceable explainer evidence manifest"
```

## Task 3: Workbench Shell, Themes, And Responsive Layout

**Files:**
- Create: `docs/explainer.html`
- Create: `docs/static/css/explainer.css`
- Modify: `tests/test_static_site.py`

**Interfaces:**
- Produces stable DOM IDs used by later tasks: `chapter-tabs`, `stage`, `inspector`, `play-toggle`, `previous-chapter`, `next-chapter`, `replay-chapter`, `tour-timeline`, `elapsed-time`, `theme-toggle`, `variant-toggle`, `before-after`, `media-dialog`, and `media-retry`.
- Consumes no JavaScript yet; controls render disabled until `explainer.js` initializes.

- [ ] **Step 1: Add failing shell and local-resource tests**

Extend `test_static_site.py` to parse `explainer.html`, require all listed IDs, require five chapter buttons, require local `explainer.css` and module `explainer.js`, and verify every local `href`, `src`, and `poster` resolves under `docs/`.

- [ ] **Step 2: Run the shell test and confirm failure**

Run: `python -m unittest discover -s tests -p 'test_static_site.py' -v`

Expected: FAIL because `docs/explainer.html` does not exist.

- [ ] **Step 3: Implement the accessible shell**

Create semantic header, `nav`, `main`, stage region, inspector `aside`, playback controls, and media `dialog`. Add an off-screen status region with `aria-live="polite"`. Each icon button receives a visible tooltip or accessible label. The page must contain a visible return link to `index.html` and no second landing-page introduction.

- [ ] **Step 4: Implement theme tokens and layout**

Define shared tokens for dark and light themes, including Base, ASRR, residual, context, focus, surface, text, muted text, and line colors. Desktop uses a stage/inspector grid inside `min-height: calc(100dvh - header)`. Mobile stacks stage above inspector and confines diagram overflow to `.stage-viewport`. Add `:focus-visible`, dialog, disabled, loading, error, and `prefers-reduced-motion` rules.

- [ ] **Step 5: Run static tests and inspect without JavaScript**

Run: `python -m unittest discover -s tests -p 'test_static_site.py' -v`

Expected: PASS. With JavaScript disabled, the page shows its title, chapter labels, stage fallback, inspector purpose, and disabled controls without overlap.

- [ ] **Step 6: Commit**

```bash
git add docs/explainer.html docs/static/css/explainer.css tests/test_static_site.py
git commit -m "Add responsive ASRR explainer workbench"
```

## Task 4: Scene Rendering And Manual Inspection

**Files:**
- Create: `docs/static/js/explainer-scenes.mjs`
- Create: `tests/explainer-controller.test.mjs`

**Interfaces:**
- Consumes: `deriveTourState`, teaching sequence, and state `{ chapterIndex, chapterProgress, variant, selectedStep, before }`.
- Produces: `renderScene(stageElement, inspectorElement, state)`, `renderChapterTabs(container, state)`, and `describeState(state)`.

- [ ] **Step 1: Write failing tests for linked identities and manual restoration**

Use a minimal fake element implementation that records assigned HTML/text. Test that the selected `a5` ID appears in both trajectory and token output, ASRR-A omits context, ASRR-C includes context, and Before restores the exact previous progress after release.

```js
test("manual variant and before state never mutate authored time", () => {
  const authored = deriveTourState(43_000);
  const manual = { ...authored, variant: "A", before: true, selectedStep: 5 };
  assert.equal(manual.timeMs, authored.timeMs);
  const restored = deriveTourState(manual.timeMs);
  assert.equal(restored.variant, authored.variant);
  assert.equal(restored.before, authored.before);
});
```

- [ ] **Step 2: Run the test and confirm missing-renderer failure**

Run: `node --test tests/explainer-controller.test.mjs`

Expected: FAIL because `explainer-scenes.mjs` does not exist.

- [ ] **Step 3: Implement Local Error and Editable Proposal scenes**

Render a shared SVG trajectory and token strip from the same stable action IDs. The first scene introduces the local-error marker and separate recorded-evidence badge. The second morphs the same action objects into tokens without replacing their IDs. Hover, click, focus, and arrow selection update both views.

- [ ] **Step 4: Implement Residual Refinement scene**

Render Base tokens, optional context, temporal refiner, per-step residual arrows, coordinate mask, bound, `alpha`, and composed output. `variant="A"` removes context paths; `variant="C"` shows only the declared global/step context. The press-and-hold comparison changes only the residual intervention and uses the same tour time.

- [ ] **Step 5: Implement Native Execution and Evidence shell scenes**

Render prediction horizon, editable prefix, native execution operator, executed prefix, and replan marker. The Evidence scene initially renders the pi0.5 pair and empty slots whose case cards later receive media from Task 5.

- [ ] **Step 6: Complete inspector copy and evidence labels**

Every chapter inspector names the takeaway, input, actual change, and evidence type. Illustrative scenes display `Illustrative teaching values`; recorded cases display `Recorded execution`; metric readouts display `Measured aggregate`.

- [ ] **Step 7: Run tests**

Run: `node --test tests/explainer-model.test.mjs tests/explainer-controller.test.mjs`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add docs/static/js/explainer-scenes.mjs tests/explainer-controller.test.mjs
git commit -m "Render linked ASRR explainer scenes"
```

## Task 5: Recorded Evidence Media Controller

**Files:**
- Create: `docs/static/js/explainer-media.mjs`
- Create: `tests/explainer-media.test.mjs`
- Modify: `docs/static/js/explainer-scenes.mjs`

**Interfaces:**
- Produces: `createMediaController({ createVideo, now })` with methods `loadCase(caseData, token)`, `sync(timeSeconds, playing)`, `pause()`, `retry()`, `dispose()`, and `getState()`.
- Consumes: validated case entries from `explainer-evidence.json` and evidence-scene card containers.

- [ ] **Step 1: Write failing cancellation, retry, and speed tests**

```js
test("stale media completion cannot restart playback", async () => {
  const harness = makeDeferredVideoHarness();
  const media = createMediaController(harness.dependencies);
  const first = media.loadCase(CASE_A, 1);
  media.dispose();
  harness.resolveLoad();
  await first;
  assert.equal(harness.playCalls, 0);
});

test("real robot media retains declared 3x playback", async () => {
  const harness = makeResolvedVideoHarness();
  const media = createMediaController(harness.dependencies);
  await media.loadCase({ ...CASE_A, playbackRate: 3 }, 1);
  media.sync(4, true);
  assert.equal(harness.video.playbackRate, 3);
});
```

Add tests for load failure exposing retry, retry clearing the error, pause freezing both videos, seek setting both current times, and invalid entries rejecting before DOM mutation.

- [ ] **Step 2: Run tests and confirm missing-controller failure**

Run: `node --test tests/explainer-media.test.mjs`

Expected: FAIL because `explainer-media.mjs` does not exist.

- [ ] **Step 3: Implement lazy paired-video management**

Create each video element once per selected case, set `preload="metadata"`, `muted`, and `playsInline`, apply the manifest playback rate on metadata and play, and derive clip time from evidence-scene progress. Use a monotonically increasing generation token around every async `load()`/`play()` operation. `dispose()` invalidates the token, pauses videos, removes listeners, and releases any object URL.

- [ ] **Step 4: Connect evidence wall and modal**

During the first Evidence interval, show the pi0.5 pair. Then scale it into the first wall position and reveal OpenVLA-OFT plus Corn-to-plate, Block-to-holder, and Block stacking. Selecting a tile pauses the tour and opens the complete Base/ASRR pair in `media-dialog`; Escape closes it and restores the authored scene.

- [ ] **Step 5: Run tests**

Run: `node --test tests/explainer-media.test.mjs tests/explainer-controller.test.mjs`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/static/js/explainer-media.mjs docs/static/js/explainer-scenes.mjs tests/explainer-media.test.mjs
git commit -m "Add synchronized explainer evidence media"
```

## Task 6: Shared Clock, Playback Controls, And Theme Persistence

**Files:**
- Create: `docs/static/js/explainer.js`
- Modify: `tests/explainer-controller.test.mjs`
- Modify: `docs/explainer.html`

**Interfaces:**
- Produces browser controller `createTourController({ model, renderer, media, storage, clock })` and clock adapter `{ getTimeMs(), play(), pause(), seek(ms), subscribe(listener) }`.
- Consumes stable shell IDs, pure state, scene renderer, and media controller.

- [ ] **Step 1: Add failing playback and restoration tests**

Test play from zero, pause, seek backward, seek to end, replay current chapter, previous/next chapter, hidden-tab pause, and Space outside form controls. Test that a manual chapter/variant/timestep/Before action pauses playback and that resuming derives authored values from the current clock time. Test invalid stored theme falling back to dark.

- [ ] **Step 2: Run controller tests and confirm failure**

Run: `node --test tests/explainer-controller.test.mjs`

Expected: FAIL because `explainer.js` does not export the controller.

- [ ] **Step 3: Implement the seekable clock and render loop**

Use `requestAnimationFrame` only while playing. Store `startedAt`, `offsetMs`, and `playing`; every frame computes clamped tour time and calls one render pass. Seeking and chapter jumps immediately derive the complete state. End at exactly `90_000` ms and switch the primary command to Replay.

- [ ] **Step 4: Bind controls and accessibility behavior**

Bind chapter tabs, play/pause, previous/next, replay, timeline, variant segmented control, timestep selection, Before pointer/Space/Enter press-and-hold, dialog Escape, and keyboard arrows. Update `aria-current`, `aria-pressed`, disabled state, elapsed time, timeline fill, document `data-stage`, and live status from shared state.

- [ ] **Step 5: Implement theme and future narration adapter boundary**

Read `asrr-explainer-theme`, default to `dark`, update `document.documentElement.dataset.theme`, and save explicit changes. Define `setExternalClock(adapter)` but do not render narration controls or audio elements; this method swaps the clock source and triggers the same state derivation.

- [ ] **Step 6: Run all Node tests**

Run: `node --test tests/explainer-*.test.mjs`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/static/js/explainer.js docs/explainer.html tests/explainer-controller.test.mjs
git commit -m "Add seekable ASRR explainer tour controls"
```

## Task 7: Homepage Entry, Mobile Fit, And Attribution

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/static/css/site.css`
- Modify: `tests/test_static_site.py`
- Create: `docs/THIRD_PARTY.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: deployed `explainer.html` and the explainer's theme/visual tokens.
- Produces: homepage preview anchor `href="explainer.html"` and documentation commands for local validation.

- [ ] **Step 1: Add failing homepage-entry and anonymity tests**

Require exactly one primary `Explore ASRR` link between the Abstract and Method section positions, require an explicit `90-second interactive tour` description, and require no lab name, real author identity, local absolute path, or new external tracking request in homepage/explainer source. Add a check that explainer images and SVG wrappers do not set a page-wide fixed minimum width above 390 px.

- [ ] **Step 2: Run static tests and confirm failure**

Run: `python -m unittest discover -s tests -p 'test_static_site.py' -v`

Expected: FAIL because the homepage entry is absent.

- [ ] **Step 3: Add the compact homepage preview**

Insert a full-width section between Abstract and Method. Use one unframed miniature trajectory: Base path, a coral residual arrow, and teal refined path. Keep the copy to one heading, one sentence, `Explore ASRR`, and `About 90 seconds`; do not create another hero or duplicate the five chapter descriptions.

- [ ] **Step 4: Document local preview and third-party references**

Add commands:

```bash
python -m http.server 8765 --directory docs
node --test tests/explainer-*.test.mjs
python -m unittest discover -s tests -v
```

`docs/THIRD_PARTY.md` records the Apache-2.0 `show-your-project-by-animation` D-JEPA reference and MIT Transformer Explainer interaction reference. State whether implementation code was adapted or only patterns were studied; preserve exact notices for any copied file.

- [ ] **Step 5: Validate desktop, mobile, light, dark, and reduced motion**

Run the local server and capture:

- homepage and explainer at 1440 x 900 in dark and light;
- explainer at 390 x 844;
- explainer at 1440 x 900 with reduced motion.

Inspect that the whole desktop workbench and controls fit, mobile has no page-wide horizontal overflow, captions and labels fit, focus is visible, and reduced motion preserves meaningful trajectory/residual state without decorative drift.

- [ ] **Step 6: Run all automated tests**

Run:

```bash
node --test tests/explainer-*.test.mjs
python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests PASS and `git diff --check` has no output.

- [ ] **Step 7: Commit**

```bash
git add docs/index.html docs/static/css/site.css docs/README.md docs/THIRD_PARTY.md tests/test_static_site.py
git commit -m "Link the ASRR interactive explainer"
```

## Task 8: Full Tour Acceptance And Deployment

**Files:**
- Modify only files whose verified acceptance issue requires correction.

**Interfaces:**
- Consumes: complete local site and all automated tests.
- Produces: verified local release, anonymous commit, pushed `main`, and successful GitHub Pages deployment.

- [ ] **Step 1: Complete a real-time tour playthrough**

Play from 0 to 90 seconds at 1x. Confirm every chapter advances without another click, important transformations hold long enough to read, the Evidence pair transitions into the wall, video time follows the shared clock, and completion ends cleanly at Replay.

- [ ] **Step 2: Exercise non-linear and failure behavior**

Seek forward/back across every chapter boundary, select action steps, switch A/C, hold/release Before, open/close a recorded case, simulate one rejected media load and retry, hide/show the tab, and replay from the end. Confirm state and controls agree after every transition.

- [ ] **Step 3: Run final verification**

Run:

```bash
node --test tests/explainer-*.test.mjs
python -m unittest discover -s tests -v
python -m compileall -q asrr_core examples tests
git diff --check
git status --short
```

Expected: Node and Python tests PASS, compileall succeeds, diff check is empty, and status contains only intended explainer changes before the final commit.

- [ ] **Step 4: Verify anonymous publication identity**

Run:

```bash
gh api user --jq .login
git config user.name
git config user.email
git remote -v
```

Expected: GitHub and Git identity are `anonymous-0525`, email is the anonymous noreply address, and the remote is `https://github.com/anonymous-0525/ASRR.git`.

- [ ] **Step 5: Commit remaining acceptance fixes**

```bash
git add -A
git commit -m "Finalize ASRR interactive explainer"
```

- [ ] **Step 6: Push and verify Pages**

Push `main`, wait for `pages-build-deployment`, then request the deployed homepage, `explainer.html`, all ES modules, JSON manifest, posters, and representative videos. Verify HTTP 200, the deployed commit SHA, visible anonymous copy, and one full online playback.

- [ ] **Step 7: Record the website-to-video handoff**

Report the deployed URLs, 90-second chapter timing, module and evidence-manifest paths, validation performed, and the deferred inputs for video production: official ICRA 2027 requirements, narration script/audio, desired interleaving points, and any additional short clips.
