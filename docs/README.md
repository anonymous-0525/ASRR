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

- `index.html`: project page content.
- `explainer.html`: 90-second interactive ASRR workbench.
- `static/css/site.css`: page styling.
- `static/css/explainer.css`: explainer themes and responsive layout.
- `static/js/site.js`: real-robot playback-rate handling.
- `static/js/explainer*.js`: tour state, rendering, playback, and evidence media.
- `static/data/explainer-evidence.json`: traceable measured values and media identities.
- `static/images/`: compressed figures used by the page.
- `static/files/asrr_paper.pdf`: anonymous paper PDF linked from the page.

## Update Notes

- Keep the page anonymous during review.
- Do not commit local filesystem paths, author names, private logs, datasets,
  checkpoints, or raw experiment outputs.
- Keep large files out of this repository. The current static page assets are
  intentionally small enough for normal GitHub Pages hosting.
- Real-robot clips are shown at 3x playback speed in the browser. The web copies
  use H.264 for browser compatibility; the source recordings are retained
  separately from this repository.
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

Open `http://localhost:8765/explainer.html`, then run both validation suites:

```bash
node --test tests/explainer-*.test.mjs
python -m unittest discover -s tests -v
```
