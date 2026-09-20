# ASRR Four-Chapter Explainer Redesign

## Purpose

Revise the existing interactive explainer so its visual story is easier to
follow, its authored sections have equal weight, and recorded evidence is shown
directly rather than represented mainly by still images. The redesign keeps the
current static GitHub Pages architecture and reuses the existing authorized
simulation and real-robot media.

The primary teaching sequence is:

1. a useful action proposal can contain one local error;
2. the proposal can be treated as an editable interface;
3. ASRR edits it with action-only or context-conditioned residual refinement;
4. paired simulation and real-robot executions provide recorded evidence.

The explainer must continue to distinguish illustrative geometry from measured
or recorded results. It must not imply online learning, browser inference, or a
runtime ambiguity detector.

## Authored Timeline

The guided tour lasts 120 seconds and contains four equal 30-second chapters:

| Time | Chapter | Primary question |
| --- | --- | --- |
| 0-30 s | Local Error | How can an otherwise useful proposal fail? |
| 30-60 s | Editable Interface | What does ASRR adapt? |
| 60-90 s | Residual Refinement | How do ASRR-A and ASRR-C change the proposal? |
| 90-120 s | Recorded Evidence | Does the same interface improve recorded execution? |

The existing Native Execution chapter is removed. Its necessary method
contract, that the base policy remains frozen and its native execution rule is
preserved, appears at the end of Residual Refinement.

## Chapter 1: Local Error

The scene enters at `a3` and advances through `a7` in order. All action labels
remain visible throughout the chapter.

- The current action is dark gold.
- Previously visited actions are pale gold.
- Future actions use a subdued neutral-gold treatment.
- Current state is also communicated through point size and outline, not color
  alone.

A simple two-link planar robot arm follows the current action point. Its base,
joints, links, and end effector are drawn in SVG, and inverse-kinematics-style
geometry places the end effector at each teaching point. Smooth interpolation
connects adjacent actions. At the declared local-error step, the arm visibly
deviates from the target region while the rest of the proposal remains useful.

The trajectory remains explicitly labelled as an illustrative action-space
projection. A recorded evidence link may remain available, but it does not
interrupt the authored sequence.

## Chapter 2: Editable Interface

The existing token-unfolding scene is replaced by a direct comparison between
policy adaptation and proposal editing.

### Policy adaptation side

Observation and instruction enter the policy, which produces an action that
drives the robot. An optimization loop around the policy communicates that
further adaptation changes the policy computation. This is a conceptual
comparison; it does not attach unsupported numerical cost claims to a specific
baseline.

### ASRR side

The frozen base policy produces a visible robot action and trajectory. The
output is labelled `Editable action proposal`, then passes through a compact
refiner before the corrected action drives the robot. The scene renders robot
motion and action-space geometry directly rather than outputting a token strip.

The staged animation first shows the common base output, then reveals the two
adaptation targets. This makes the distinction explicit: conventional
adaptation modifies the policy, while ASRR modifies the action proposal it has
already produced.

## Chapter 3: Residual Refinement

The chapter is divided evenly between two authored modes:

- `ASRR-A`, which conditions only on the base action proposal;
- `ASRR-C`, which additionally receives available policy-side information such
  as state, trajectory mode, or action-generation features.

Both modes use one stable visual comparison:

- a translucent arm follows the gold Base trajectory;
- an opaque arm follows the teal Refined trajectory;
- coral curved arrows connect Base positions to their refined positions;
- the active action and its residual remain linked by identity;
- local geometric differences are enlarged enough to teach the operation and
  are labelled as illustrative, not to scale.

In ASRR-A, residual arrows are generated from the proposal alone. In ASRR-C,
policy-data blocks feed the refiner, and the curved residual arrows appear as
the context-conditioned correction is applied at each active action. The
existing ASRR-A/ASRR-C manual control and press-and-hold Base comparison remain
available.

The closing state displays `Base policy frozen` and `Native execution
preserved`, absorbing the only essential message from the removed execution
chapter.

## Chapter 4: Recorded Evidence

The automatic tour replaces the static evidence wall with two paired video
comparisons:

1. `pi0.5 / LIBERO-10`, Base failure and ASRR success, synchronized side by
   side;
2. `Piper / Corn-to-plate`, Base failure and ASRR success, synchronized side by
   side at the declared 3x playback speed.

The first pair occupies the first half of the chapter and the second pair the
second half. Base and ASRR labels remain fixed and unambiguous. The video clock
is derived from the shared tour clock so seeking, pausing, replaying, and tab
visibility affect both clips consistently.

Measured results appear as a restrained overlay rather than a separate static
wall: `+14.4 pp` pi0.5 mean gain, `25-63%` less recorded compute in the stated
comparison, and `+6.7 pp` aggregate real-robot gain. OpenVLA-OFT and the other
two Piper tasks remain accessible through a secondary `More evidence` command,
but do not interrupt the automatic four-chapter narrative.

Media failures retain the existing poster, retry control, and schematic
fallback. The paired videos must remain usable when browser autoplay is
blocked.

## Homepage Integration

The full interactive explainer becomes the first substantive experience below
the project title and summary. The current Abstract-to-Method preview is
removed. The embedded and standalone presentations reuse the same timeline,
scene renderers, evidence manifest, and media controller; only the surrounding
shell and control density differ.

The homepage version begins silent playback when it enters the viewport and
pauses when it leaves. It includes a command to open the full workbench without
duplicating scene logic.

The static hero mosaic is replaced by a two-by-two moving media mosaic:

- pi0.5 Base;
- pi0.5 ASRR;
- real-robot Base;
- real-robot ASRR.

These clips are muted, looped, `playsinline`, and visually labelled. They use
poster fallbacks and pause when hidden. Under reduced-motion preferences, the
hero shows the existing posters instead of autoplaying video.

## Architecture

The current modules remain the ownership boundaries:

- `explainer-model.mjs` owns the 120-second timeline, sequential action state,
  variant state, robot geometry inputs, and deterministic teaching values;
- `explainer-scenes.mjs` owns the four chapter renderers and SVG robot arms;
- `explainer-media.mjs` owns synchronized evidence playback, playback rates,
  failure recovery, and cleanup;
- `explainer.js` owns the shared controller and initializes standalone or
  embedded shells;
- `explainer-evidence.json` owns media paths, posters, playback rates, labels,
  and optional clip offsets;
- `explainer.css` and `site.css` own standalone and homepage responsive
  presentation respectively.

No framework migration, remote API, model download, or new runtime dependency
is introduced. A reusable explainer root is preferred over an iframe so the
homepage and standalone page share accessibility, theme, and playback state
without nesting documents.

## Responsive And Accessibility Requirements

- Desktop keeps the stage, inspector, and playback controls visible without
  chapter-level vertical overflow.
- Mobile stacks the stage and explanatory copy; wide comparison media scrolls
  only inside its own region when necessary.
- All four chapters remain navigable without autoplay.
- Current, visited, and future actions use shape and stroke cues in addition to
  gold shades.
- Videos are muted by default and retain explicit Base/ASRR labels.
- Keyboard navigation, focus visibility, reduced motion, hidden-tab pause, and
  theme persistence remain supported.

## Verification

Verification will cover:

1. pure timeline tests for four equal chapters and sequential `a0` through
   `a7` action progression;
2. scene tests for action history classes, both residual modes, arm identity,
   and the merged execution contract;
3. media tests for synchronized simulation playback, 3x physical playback,
   retry, stale-load cancellation, and visibility pause;
4. static-site tests for the embedded explainer, removal of the old preview,
   anonymous local assets, and the dynamic hero mosaic;
5. complete 120-second desktop and mobile playthroughs, arbitrary seek,
   chapter navigation, manual ASRR-A/ASRR-C selection, Base hold, reduced
   motion, and blocked-autoplay behavior;
6. GitHub Pages subpath and MIME verification after publication.

## Deferred Video Production

The revised 120-second timeline remains a source for the later ICRA video, not
the final submitted video itself. Narration, additional physical clips, export
settings, and official ICRA requirements remain a separately reviewed phase.
