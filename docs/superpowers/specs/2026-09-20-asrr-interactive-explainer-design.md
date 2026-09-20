# ASRR Interactive Explainer Design

## Purpose

Build a focused interactive explanation of ASRR that helps an ICRA audience
follow one action proposal through local error, residual correction, native
execution, and measured evidence. The explainer will be a separate static page
linked prominently from the existing anonymous project homepage.

The first release is a silent, approximately 90-second guided tour. Its scene
state and timing will later become the visual backbone of an ICRA video. That
video may interleave the guided animation with real-robot footage and
author-supplied narration, but video production is outside this implementation
phase. Before producing that video, the team will verify the then-current
official ICRA 2027 video, anonymity, copyright, format, and duration rules.

## Audience And Success Criteria

The primary audience is ICRA reviewers and robotics researchers. A successful
viewer should understand, without reading the full paper, that:

1. a frozen policy can already produce a useful but locally inaccurate action
   sequence;
2. ASRR treats that sequence as an editable adaptation interface;
3. a compact refiner predicts bounded, masked residuals, optionally using
   policy-side context;
4. the refined proposal retains the base controller's native execution rule;
5. simulation, VLA, compute, and physical-robot evidence support the method.

The explainer must not imply online learning, runtime ambiguity detection,
browser model inference, or experiments beyond those reported in the paper.

## Page Architecture

### Main project page

The existing homepage remains a concise paper page. Between the Abstract and
Method sections it gains a compact explainer preview with:

- one clear ASRR transformation visual;
- a primary `Explore ASRR` command;
- a short statement that the interactive tour takes about 90 seconds.

The existing results, analysis, simulation videos, real-robot videos, code, and
citation sections remain in place. The preview is not a second large hero.

### Interactive page

`docs/explainer.html` is a dedicated workbench rather than another landing
page. It contains:

- a compact anonymous ASRR header with a return link;
- five visible chapter tabs;
- a central SVG/DOM stage;
- an inspector for the current operation, formula, and evidence identity;
- one shared playback strip with play/pause, previous, next, replay, timeline,
  elapsed time, and theme control.

The desktop target is 1440 x 900 with the entire workbench and controls visible.
At mobile widths, the stage stacks above the inspector and genuinely wide
diagrams may scroll inside their own viewport.

## Explanation Contract

### 1. Local Error

- **Takeaway:** broadly correct behavior can still fail because of a localized
  action error.
- **Object identity:** one illustrative future action sequence with stable
  timestep IDs.
- **Change:** a selected action deviates from the useful trajectory near task
  completion.
- **Interaction:** select or scrub timesteps; reveal the local error and compare
  it with the rest of the proposal.
- **Evidence:** the trajectory is explicitly labelled as an illustrative
  action-space projection; a recorded Base failure is separately labelled as
  real execution evidence.

### 2. Editable Proposal

- **Takeaway:** the predicted sequence is a structured interface available
  between policy computation and physical execution.
- **Input:** the frozen policy's proposal `A_base`.
- **Change:** the proposal unfolds into temporally ordered action tokens while
  preserving timestep identity across the trajectory and token strip.
- **Interaction:** select a token and inspect its action-space value and temporal
  position.
- **Evidence:** definitions follow Eqs. (1) and (3) of the paper; displayed
  numerical values are deterministic teaching values, not exported activations.

### 3. Residual Refinement

- **Takeaway:** ASRR predicts a conservative sequence residual while the base
  policy remains frozen.
- **Input:** `A_base`, with optional global or step-aligned policy context.
- **Change:** temporal encoding produces `Delta A`; mask `M`, residual scale
  `alpha`, and a coordinate bound control where and how strongly the sequence is
  edited.
- **Output:** `A_refined = A_base + alpha * (M * Delta A)`.
- **Interaction:** switch ASRR-A/ASRR-C, select a timestep, and hold
  Before/After to inspect the declared intervention only.
- **Evidence:** context and residual values are deterministic teaching data. The
  interface and formula match the paper; the page does not claim a browser
  forward pass through a trained checkpoint.

### 4. Native Execution

- **Takeaway:** ASRR edits the proposal and preserves the base controller's
  execution and replanning protocol.
- **Input:** refined sequence plus the base execution operator.
- **Change:** editable and retained timesteps are distinguished, then an
  executable prefix is consumed before replanning.
- **Interaction:** scrub the proposal-to-execution transition and compare direct
  prefix consumption with the abstract native execution operator.
- **Evidence:** the scene explains the method contract and does not turn horizon
  choice into a separate contribution.

### 5. Evidence

- **Takeaway:** the same output-space interface improves different policy
  families and physical tasks.
- **Opening evidence:** a recorded pi0.5 Base failure and ASRR success pair.
- **Expansion:** the main pair shrinks into an evidence wall containing
  OpenVLA-OFT and the three Piper tasks. Full clips remain available in a modal
  or via the existing page sections.
- **Measured readouts:** `+14.4 pp` pi0.5 mean gain, `62.0 -> 80.9` at 5k,
  `25-63%` lower recorded cache-and-fitting GPU-hours in the stated comparison,
  and `+6.7 pp` aggregate real-robot gain.
- **Interaction:** select a case, replay its recorded comparison, and return to
  the summary wall.
- **Evidence identity:** recorded clips, aggregate metrics, and illustrative
  computation remain visually and textually distinct.

## Visual System

The explainer reuses the paper's semantic roles rather than the bundled
reference palette:

- Base/frozen proposal: muted blue;
- ASRR/refined result: teal;
- residual/local edit: coral;
- optional context: lavender;
- neutral structure: gray;
- success/failure labels: accessible teal and muted coral variants.

Dark is the initial theme because trajectories and residual motion read clearly
against it. A light theme is available and the selected preference is stored
locally. Typography follows the existing Inter-based project page with a small,
stable hierarchy. The stage uses restrained SVG motion, not decorative 3D or
perpetually moving backgrounds.

## Interaction And Timing

The silent tour lasts approximately 90 seconds across five authored chapter
intervals. One shared monotonic clock determines chapter, progress, selected
object, trajectory geometry, captions, video time, and holds. Rendering never
owns an independent timeout chain.

Required controls:

- Play/Pause;
- Previous/Next chapter;
- Replay current chapter;
- arbitrary timeline seek;
- chapter navigation;
- ASRR-A/ASRR-C selection in the refinement chapter;
- timestep selection;
- press-and-hold Before/After comparison;
- dark/light theme toggle.

Seeking restores all derived state. Changing a manual control pauses guided
playback; resuming returns to the authored state at the current tour time.
Reduced-motion mode removes decorative transitions while preserving meaningful
state changes.

No narration control is shown in the first release. The orchestration layer
will expose a clock adapter so later audio time can drive the same state,
captions, and intentional holds without rewriting scene logic.

## Technical Structure

The implementation remains static and compatible with the existing GitHub
Pages deployment:

- `docs/explainer.html`: workbench shell and accessible controls;
- `docs/static/css/explainer.css`: theme tokens, responsive layout, focus, and
  reduced-motion behavior;
- `docs/static/js/explainer.js`: orchestration, playback, events, and rendering
  coordination;
- `docs/static/js/explainer-model.mjs`: pure deterministic teaching values,
  ASRR composition, masks, bounds, and timeline state;
- `docs/static/js/explainer-scenes.mjs`: chapter renderers and linked SVG views;
- `docs/static/js/explainer-media.mjs`: lazy video loading, synchronized seek,
  pause, retry, and cleanup;
- `docs/static/data/explainer-evidence.json`: metric values, media identities,
  labels, playback speeds, and evidence notes.

The homepage reuses existing files and receives only the preview markup, styles,
and entry link. No framework migration, online API, research weights, model
download, or browser inference is introduced.

The implementation may adapt the state, playback, theme, and testing patterns
from the Apache-2.0 D-JEPA reference and MIT Transformer reference bundled with
`show-your-project-by-animation`. Any copied implementation must preserve the
applicable attribution and license notices. Scientific content, data, palette,
and scene logic will be ASRR-specific.

## Media Handling And Failure States

Existing authorized project media is reused. Full videos remain unchanged and
retain their declared playback speeds; lightweight preview media may be derived
for the evidence wall while retaining links to complete clips. Cropping must not
hide the robot, manipulated object, goal, or terminal outcome.

Media loads on demand. A failed clip shows its poster, an explicit unavailable
state, and a retry control. The tour continues through schematic scenes if media
cannot load. Stale play promises are invalidated when the user pauses, seeks,
changes chapter, or leaves the page. The page pauses when hidden.

## Validation

Validation includes:

1. Node tests for pure ASRR composition, mask/bound behavior, chapter timing,
   arbitrary seek, and evidence-manifest invariants.
2. Existing Python static-site tests extended for the explainer entry, local
   resources, anonymity, and media scope.
3. Local HTTP inspection in dark and light themes at 1440 x 900 and 390 px.
4. A complete 90-second playthrough plus targeted tests for seek, chapter jumps,
   Before/After restoration, ASRR-A/ASRR-C state, media retry, reduced motion,
   keyboard focus, and hidden-tab pause.
5. GitHub Pages subpath verification after authorized publication.

## Deferred ICRA Video Phase

After the website explainer is accepted, video production will begin as a
separate scoped task. The 90-second tour provides the central explanatory
sequence. Additional physical-robot clips and author-supplied narration can be
interleaved around or within that sequence. Before editing or export, the team
will retrieve and follow the current official ICRA 2027 submission requirements
instead of assuming a duration, codec, resolution, file size, anonymity policy,
or copyright rule.
