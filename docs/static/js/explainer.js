import {
  CHAPTERS,
  TOUR_DURATION_MS,
  deriveTourState,
} from "./explainer-model.mjs";
import {
  describeState,
  renderChapterTabs,
  renderScene,
} from "./explainer-scenes.mjs";
import { createMediaController } from "./explainer-media.mjs";

const THEME_KEY = "asrr-explainer-theme";
const THEMES = new Set(["dark", "light"]);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function createTourController({ model, renderer, media, storage, clock }) {
  if (!model || typeof model.deriveTourState !== "function") {
    throw new TypeError("Tour controller requires a deriveTourState model.");
  }
  if (typeof renderer !== "function") {
    throw new TypeError("Tour controller requires a renderer.");
  }
  if (!clock) throw new TypeError("Tour controller requires a clock adapter.");

  let activeClock = clock;
  let unsubscribe = null;
  let playing = Boolean(activeClock.isPlaying?.());
  let manualOverrides = null;
  const storedTheme = storage?.getItem?.(THEME_KEY);
  let theme = THEMES.has(storedTheme) ? storedTheme : "dark";
  let state = null;

  function render(timeMs = activeClock.getTimeMs()) {
    const scene = model.deriveTourState(timeMs, manualOverrides ?? {});
    if (scene.timeMs >= TOUR_DURATION_MS && playing) {
      playing = false;
      activeClock.pause();
    }
    state = {
      timeMs: scene.timeMs,
      playing,
      atEnd: scene.timeMs >= TOUR_DURATION_MS,
      theme,
      scene,
      manual: manualOverrides !== null,
    };
    renderer(scene, state);
    if (scene.chapterIndex === 3) {
      media?.sync?.(scene.evidenceLocalTimeMs / 1000, playing);
    } else {
      media?.pause?.();
    }
    return state;
  }

  function attachClock(nextClock) {
    unsubscribe?.();
    activeClock = nextClock;
    playing = Boolean(activeClock.isPlaying?.());
    unsubscribe = activeClock.subscribe?.((timeMs, clockPlaying) => {
      if (typeof clockPlaying === "boolean") playing = clockPlaying;
      render(timeMs);
    }) ?? null;
    render(activeClock.getTimeMs());
  }

  function pause() {
    playing = false;
    activeClock.pause();
    media?.pause?.();
    render();
  }

  function play() {
    if (state?.atEnd) activeClock.seek(0);
    manualOverrides = null;
    playing = true;
    activeClock.play();
    render();
  }

  function seek(timeMs) {
    pause();
    manualOverrides = null;
    activeClock.seek(clamp(timeMs, 0, TOUR_DURATION_MS));
    render();
  }

  function setManual(patch) {
    pause();
    manualOverrides = { ...(manualOverrides ?? {}), ...patch };
    render();
  }

  function setVariant(variant) {
    if (!new Set(["A", "C"]).has(variant)) return;
    setManual({ variant });
  }

  function selectStep(selectedStep) {
    if (!Number.isInteger(selectedStep)) return;
    setManual({ selectedStep });
  }

  function setBefore(before) {
    setManual({ before: Boolean(before) });
  }

  function chapterJump(index) {
    const chapter = CHAPTERS[clamp(index, 0, CHAPTERS.length - 1)];
    seek(chapter.startMs);
  }

  function nextChapter() {
    chapterJump((state?.scene.chapterIndex ?? 0) + 1);
  }

  function previousChapter() {
    chapterJump((state?.scene.chapterIndex ?? 0) - 1);
  }

  function replayChapter() {
    chapterJump(state?.scene.chapterIndex ?? 0);
  }

  function setTheme(nextTheme) {
    if (!THEMES.has(nextTheme)) return;
    theme = nextTheme;
    storage?.setItem?.(THEME_KEY, theme);
    render();
  }

  function onVisibilityChange(hidden) {
    if (hidden && playing) pause();
  }

  function handleKey(event) {
    const tagName = event.target?.tagName?.toUpperCase?.() ?? "";
    if (new Set(["INPUT", "BUTTON", "TEXTAREA", "SELECT", "VIDEO"]).has(tagName)) return;
    if (event.code === "Space") {
      event.preventDefault?.();
      if (playing) pause(); else play();
    }
  }

  function setExternalClock(adapter) {
    if (!adapter || typeof adapter.getTimeMs !== "function") {
      throw new TypeError("External clock must implement the clock adapter.");
    }
    activeClock.pause?.();
    manualOverrides = null;
    attachClock(adapter);
  }

  function dispose() {
    unsubscribe?.();
    activeClock.pause?.();
    media?.pause?.();
  }

  attachClock(activeClock);

  return {
    play,
    pause,
    seek,
    nextChapter,
    previousChapter,
    replayChapter,
    setVariant,
    selectStep,
    setBefore,
    setTheme,
    setExternalClock,
    onVisibilityChange,
    handleKey,
    getState: () => state,
    dispose,
  };
}

export function createRafClock({
  durationMs = TOUR_DURATION_MS,
  now = () => performance.now(),
  requestFrame = (callback) => requestAnimationFrame(callback),
  cancelFrame = (handle) => cancelAnimationFrame(handle),
} = {}) {
  let timeMs = 0;
  let playing = false;
  let origin = 0;
  let frameHandle = null;
  const listeners = new Set();

  const emit = () => listeners.forEach((listener) => listener(timeMs, playing));
  const frame = () => {
    if (!playing) return;
    timeMs = clamp(now() - origin, 0, durationMs);
    if (timeMs >= durationMs) playing = false;
    emit();
    if (playing) frameHandle = requestFrame(frame);
  };

  return {
    getTimeMs: () => timeMs,
    isPlaying: () => playing,
    play() {
      if (playing) return;
      if (timeMs >= durationMs) timeMs = 0;
      playing = true;
      origin = now() - timeMs;
      emit();
      frameHandle = requestFrame(frame);
    },
    pause() {
      if (!playing) return;
      playing = false;
      if (frameHandle !== null) cancelFrame(frameHandle);
      frameHandle = null;
      emit();
    },
    seek(nextTimeMs) {
      timeMs = clamp(nextTimeMs, 0, durationMs);
      if (playing) origin = now() - timeMs;
      emit();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

function formatTime(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function browserVideoFactory(documentRef, { role, src, poster }) {
  const video = documentRef.createElement("video");
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = src;
  video.poster = poster;
  video.dataset.role = role;
  return video;
}

export function mountMediaPair(container, videos, documentRef = document) {
  container.replaceChildren();
  for (const [label, video] of [["Base", videos[0]], ["ASRR", videos[1]]]) {
    if (!video) continue;
    const figure = documentRef.createElement("figure");
    const caption = documentRef.createElement("figcaption");
    caption.textContent = label;
    figure.append(video, caption);
    container.append(figure);
  }
}

export function bindBeforeAfterRelease(documentRef, controller) {
  const release = () => {
    if (controller.getState()?.scene.before) controller.setBefore(false);
  };
  const releaseKey = (event) => {
    if (new Set(["Space", "Enter"]).has(event.code)) release();
  };
  documentRef.addEventListener("pointerup", release);
  documentRef.addEventListener("pointercancel", release);
  documentRef.addEventListener("keyup", releaseKey);
  return () => {
    documentRef.removeEventListener?.("pointerup", release);
    documentRef.removeEventListener?.("pointercancel", release);
    documentRef.removeEventListener?.("keyup", releaseKey);
  };
}

export async function mountExplainer(root, {
  mode = "standalone",
  autoplayWhenVisible = false,
  fetchImpl,
} = {}) {
  if (!root || typeof root.querySelector !== "function") {
    throw new TypeError("mountExplainer requires a root element.");
  }
  if (!new Set(["standalone", "embedded"]).has(mode)) {
    throw new RangeError("Explainer mode must be standalone or embedded.");
  }
  root.dataset.explainerMode = mode;
  const documentRef = root.ownerDocument ?? document;
  const windowRef = documentRef.defaultView ?? window;
  const query = (id) => root.querySelector(`#${id}`);
  const elements = {
    tabs: query("chapter-tabs"),
    stage: query("stage"),
    inspector: query("inspector"),
    play: query("play-toggle"),
    previous: query("previous-chapter"),
    next: query("next-chapter"),
    replay: query("replay-chapter"),
    timeline: query("tour-timeline"),
    elapsed: query("elapsed-time"),
    theme: query("theme-toggle"),
    dialog: query("media-dialog"),
    mediaPair: query("media-pair"),
    mediaStatus: query("media-status"),
    retry: query("media-retry"),
    live: query("tour-status"),
    heading: query("stage-heading"),
  };

  const missing = Object.entries(elements).filter(([, element]) => !element).map(([name]) => name);
  if (missing.length) throw new Error(`Explainer root is missing: ${missing.join(", ")}.`);
  elements.timeline.max = String(TOUR_DURATION_MS);
  root.querySelectorAll(".timeline-chapters i").forEach((marker, index) => {
    marker.style.left = `${100 * CHAPTERS[index + 1].startMs / TOUR_DURATION_MS}%`;
  });

  const fetcher = fetchImpl ?? windowRef.fetch.bind(windowRef);
  const response = await fetcher("static/data/explainer-evidence.json");
  if (!response.ok) throw new Error(`Evidence manifest failed with ${response.status}.`);
  const evidence = await response.json();
  const evidenceById = new Map(evidence.cases.map((item) => [item.id, item]));
  const createVideo = (options) => browserVideoFactory(documentRef, options);
  const authoredMedia = createMediaController({ createVideo, now: () => windowRef.performance.now() });
  const modalMedia = createMediaController({ createVideo, now: () => windowRef.performance.now() });
  const clock = createRafClock({
    now: () => windowRef.performance.now(),
    requestFrame: (callback) => windowRef.requestAnimationFrame(callback),
    cancelFrame: (handle) => windowRef.cancelAnimationFrame(handle),
  });
  const cleanups = [];
  let userPaused = false;
  const listen = (target, type, listener, options) => {
    target.addEventListener(type, listener, options);
    cleanups.push(() => target.removeEventListener(type, listener, options));
  };
  let renderedEvidenceCase = null;
  let authoredCaseId = null;

  async function ensureAuthoredEvidence(scene) {
    const caseData = evidenceById.get(scene.evidenceCaseId);
    const host = elements.stage.querySelector("[data-authored-media-pair]");
    if (!caseData || !host) return;

    if (authoredCaseId !== caseData.id) {
      authoredCaseId = caseData.id;
      const token = authoredMedia.getState().generation + 1;
      const loadPromise = authoredMedia.loadPair(caseData, token);
      mountMediaPair(host, authoredMedia.getVideos());
      const result = await loadPromise;
      if (authoredCaseId !== caseData.id || result.status !== "ready") return;
      const currentHost = elements.stage.querySelector("[data-authored-media-pair]");
      if (currentHost) mountMediaPair(currentHost, authoredMedia.getVideos());
      authoredMedia.sync(scene.evidenceLocalTimeMs / 1000, clock.isPlaying());
      return;
    }

    if (!host.children.length) mountMediaPair(host, authoredMedia.getVideos());
  }

  function updateChrome(scene, controllerState) {
    const evidenceChanged = scene.chapterIndex === 3 && renderedEvidenceCase !== scene.evidenceCaseId;
    if (scene.chapterIndex !== 3 || evidenceChanged) {
      renderScene(elements.stage, elements.inspector, scene, { homePath: mode === "standalone" ? "./" : "" });
      renderedEvidenceCase = scene.chapterIndex === 3 ? scene.evidenceCaseId : null;
      if (scene.chapterIndex === 3) ensureAuthoredEvidence(scene);
    }
    renderChapterTabs(elements.tabs, scene);
    elements.timeline.value = String(scene.timeMs);
    elements.elapsed.textContent = `${formatTime(scene.timeMs)} / ${formatTime(TOUR_DURATION_MS)}`;
    elements.play.innerHTML = controllerState.atEnd
      ? '<span aria-hidden="true">&#8634;</span><span>Replay</span>'
      : controllerState.playing
        ? '<span aria-hidden="true">&#10074;&#10074;</span><span>Pause</span>'
        : '<span aria-hidden="true">&#9654;</span><span>Play</span>';
    elements.play.setAttribute("aria-label", controllerState.atEnd ? "Replay guided tour" : controllerState.playing ? "Pause guided tour" : "Play guided tour");
    elements.previous.disabled = scene.chapterIndex === 0;
    elements.next.disabled = scene.chapterIndex === CHAPTERS.length - 1;
    elements.heading.textContent = scene.chapter.label;
    elements.live.textContent = describeState(scene);
    const themeTarget = mode === "standalone" ? documentRef.documentElement : root;
    themeTarget.dataset.theme = controllerState.theme;
    root.dataset.stage = scene.chapter.id;
    elements.theme.innerHTML = controllerState.theme === "dark" ? '<span aria-hidden="true">&#9788;</span>' : '<span aria-hidden="true">&#9790;</span>';
    elements.theme.setAttribute("aria-label", controllerState.theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  const controller = createTourController({
    model: { deriveTourState },
    renderer: updateChrome,
    media: authoredMedia,
    storage: windowRef.localStorage,
    clock,
  });
  cleanups.push(bindBeforeAfterRelease(documentRef, controller));

  for (const control of [elements.play, elements.replay, elements.timeline, elements.theme]) control.disabled = false;

  listen(elements.play, "click", () => {
    if (controller.getState().playing) {
      userPaused = true;
      controller.pause();
    } else {
      userPaused = false;
      controller.play();
    }
  });
  listen(elements.previous, "click", controller.previousChapter);
  listen(elements.next, "click", controller.nextChapter);
  listen(elements.replay, "click", controller.replayChapter);
  listen(elements.timeline, "input", (event) => controller.seek(Number(event.target.value)));
  listen(elements.theme, "click", () => controller.setTheme(controller.getState().theme === "dark" ? "light" : "dark"));
  listen(elements.tabs, "click", (event) => {
    const button = event.target.closest("[data-chapter-index]");
    if (!button) return;
    controller.seek(CHAPTERS[Number(button.dataset.chapterIndex)].startMs);
  });
  listen(elements.stage, "click", (event) => {
    const action = event.target.closest("[data-step]");
    if (action) controller.selectStep(Number(action.dataset.step));
  });
  listen(elements.stage, "keydown", (event) => {
    const action = event.target.closest?.("[data-step]");
    if (!action || !new Set(["ArrowLeft", "ArrowRight"]).has(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    controller.selectStep(clamp(Number(action.dataset.step) + offset, 0, 7));
  });
  listen(elements.inspector, "click", (event) => {
    const variant = event.target.closest("[data-variant]");
    if (variant) controller.setVariant(variant.dataset.variant);
  });
  listen(elements.inspector, "pointerdown", (event) => {
    if (event.target.closest("#before-after")) controller.setBefore(true);
  });
  for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
    listen(elements.inspector, eventName, (event) => {
      if (event.target.closest?.("#before-after") || controller.getState().scene.before) controller.setBefore(false);
    });
  }
  listen(elements.inspector, "keydown", (event) => {
    if (!event.target.closest?.("#before-after") || !new Set(["Space", "Enter"]).has(event.code)) return;
    event.preventDefault();
    event.stopPropagation();
    controller.setBefore(true);
  });
  listen(elements.inspector, "keyup", (event) => {
    if (!event.target.closest?.("#before-after") || !new Set(["Space", "Enter"]).has(event.code)) return;
    event.preventDefault();
    event.stopPropagation();
    controller.setBefore(false);
  });

  async function openEvidence(caseId) {
    const caseData = evidenceById.get(caseId);
    if (!caseData) return;
    controller.pause();
    elements.mediaPair.replaceChildren();
    elements.mediaStatus.textContent = `Loading ${caseData.policy} / ${caseData.task}...`;
    elements.retry.hidden = true;
    if (!elements.dialog.open) elements.dialog.showModal();
    const loadPromise = modalMedia.loadPair(caseData, modalMedia.getState().generation + 1);
    mountMediaPair(elements.mediaPair, modalMedia.getVideos());
    const mediaState = await loadPromise;
    if (mediaState.status === "error") {
      elements.mediaStatus.textContent = "Recorded media is unavailable. The schematic tour remains active.";
      elements.retry.hidden = false;
    } else {
      elements.mediaStatus.textContent = `${caseData.note} ${caseData.playbackRate === 3 ? "Playback is set to 3x." : ""}`;
      modalMedia.sync(0, false);
    }
  }

  listen(root, "click", (event) => {
    if (event.target.closest?.("[data-page-section]")) {
      userPaused = true;
      controller.pause();
      return;
    }
    const evidenceCase = event.target.closest?.("[data-evidence-case]");
    if (evidenceCase) openEvidence(evidenceCase.dataset.evidenceCase);
    if (event.target.closest?.("[data-dialog-close]")) elements.dialog.close();
  });
  listen(elements.dialog, "close", () => modalMedia.pause());
  listen(elements.retry, "click", async () => {
    elements.retry.hidden = true;
    elements.mediaStatus.textContent = "Retrying recorded media...";
    const retryPromise = modalMedia.retry();
    mountMediaPair(elements.mediaPair, modalMedia.getVideos());
    const result = await retryPromise;
    elements.retry.hidden = result.status !== "error";
    elements.mediaStatus.textContent = result.status === "ready" ? result.caseData.note : "Recorded media is still unavailable.";
  });
  listen(documentRef, "visibilitychange", () => controller.onVisibilityChange(documentRef.hidden));
  listen(documentRef, "keydown", (event) => controller.handleKey(event));

  let observer = null;
  if (autoplayWhenVisible && typeof windowRef.IntersectionObserver === "function") {
    observer = new windowRef.IntersectionObserver((entries) => {
      const entry = entries.find((item) => item.target === root);
      if (!entry) return;
      if (entry.intersectionRatio >= 0.55 && !userPaused && !controller.getState().playing) {
        controller.play();
      } else if (entry.intersectionRatio < 0.2 && controller.getState().playing) {
        controller.pause();
      }
    }, { threshold: [0, 0.2, 0.55, 1] });
    observer.observe(root);
  }

  function destroy() {
    observer?.disconnect();
    for (const cleanup of cleanups.splice(0)) cleanup();
    controller.dispose();
    authoredMedia.dispose();
    modalMedia.dispose();
  }

  listen(windowRef, "pagehide", destroy, { once: true });
  return { controller, destroy };
}

if (typeof document !== "undefined") {
  const root = document.querySelector('[data-explainer-root="standalone"][data-explainer-auto]');
  if (root) {
    mountExplainer(root, { mode: "standalone" }).catch((error) => {
      const status = root.querySelector("#tour-status");
      if (status) status.textContent = `Explainer could not initialize: ${error.message}`;
    });
  }
}
