# ASRR Project Page

GitHub Pages is configured to serve this directory from:

```text
main /docs
```

The live page is:

```text
https://anonymous-0525.github.io/ASRR/
```

## Files

- `index.html`: project page content with the live explainer embedded directly below the homepage hero.
- `explainer.html`: standalone 105-second, four-chapter ASRR workbench.
- `static/images/explainer/arm/`: original Blender-rendered parts animated in 2D; regenerate with `blender --background --python tools/render_explainer_arm.py` from the repository root.
- `static/css/site.css`: page styling.
- `static/css/explainer.css`: explainer themes and responsive layout.
- `static/js/site.js`: moving hero media, visibility-aware playback, and homepage explainer mounting.
- `static/js/explainer*.js`: tour state, rendering, playback, and evidence media.
- `static/data/explainer-evidence.json`: traceable measured values and media identities.
- `static/images/`: compressed figures used by the page.
- `static/files/asrr_paper.pdf`: anonymous paper PDF linked from the page.
- `../tools/export_explainer.mjs`: captures the clean presentation as a silent
  1080p/30fps MP4 and four chapter clips using Firefox, Xvfb, and FFmpeg.

## Update Notes

- Keep the page anonymous during review.
- Do not commit local filesystem paths, author names, private logs, datasets,
  checkpoints, or raw experiment outputs.
- Keep large files out of this repository. The current static page assets are
  intentionally small enough for normal GitHub Pages hosting.
- Real-robot clips are shown at 3x playback speed in the browser. The web copies
  use H.264 for browser compatibility; the source recordings are retained
  separately from this repository.
- The first three chapters last 25 seconds each: Local Error, Editable Interface,
  and Residual Refinement. Recorded Evidence retains 30 seconds, presenting
  pi0.5 / LIBERO-10 followed by Piper / Corn-to-plate.
- The blue ideal, yellow Base, and green refined trajectories are distinct.
  The Base approaches but misses the target; refinement reaches the target with
  a small remaining offset from the ideal. Transparent arms follow the ideal in
  the first two chapters and the Base in the residual comparison. Recorded
  Evidence links directly to the homepage videos and results.
- Homepage videos retain their original aspect ratios and show captions outside
  the picture. The green Interactive Demo link opens `explainer.html?autoplay=1`.
- When refreshing figures or the paper PDF, run the standard-library tests from
  the repository root:

```bash
python -m unittest discover -s tests
```

## Local explainer preview

From the repository root:

```bash
python -m http.server 8765 --directory docs
```

Open `http://localhost:8765/` for the embedded workbench or
`http://localhost:8765/explainer.html` for the full-screen version, then run
both validation suites:

```bash
node --test tests/explainer-*.test.mjs
python -m unittest discover -s tests -v
```

## Video material export

`explainer.html?export=1` uses the same timeline and recordings, with navigation,
playback buttons, interactive controls, and native video controls hidden. It
starts paused for capture. The export is editing material, without narration;
it is not a finished conference submission video.

Run the local server above, then open the export URL in a 1920x1080 Firefox kiosk
session on an isolated Xvfb display. With the WebDriver session ID:

```bash
node tools/export_explainer.mjs \
  --session WEBDRIVER_SESSION_ID \
  --display :97 \
  --output artifacts/explainer-video
```

FFmpeg and ffprobe must be on `PATH`, or specified via `--ffmpeg` and `--ffprobe`.
The output includes a 105-second master, four chapter MP4s (25/25/25/30 seconds),
inspection PNGs, and format metadata. Physical recordings retain 3x playback.
Keep generated video exports outside the public site assets.
