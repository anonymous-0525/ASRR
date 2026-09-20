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
    if (scene.chapterIndex === 4) {
      media?.sync?.(Math.max(0, (scene.timeMs - CHAPTERS[4].startMs) / 1000), playing);
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

function browserVideoFactory({ role, src, poster }) {
  const video = document.createElement("video");
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
  documentRef.addEventListener("pointerup", release);
  documentRef.addEventListener("pointercancel", release);
  documentRef.addEventListener("keyup", (event) => {
    if (new Set(["Space", "Enter"]).has(event.code)) release();
  });
}

async function initializeBrowserExplainer() {
  const elements = {
    tabs: document.getElementById("chapter-tabs"),
    stage: document.getElementById("stage"),
    inspector: document.getElementById("inspector"),
    play: document.getElementById("play-toggle"),
    previous: document.getElementById("previous-chapter"),
    next: document.getElementById("next-chapter"),
    replay: document.getElementById("replay-chapter"),
    timeline: document.getElementById("tour-timeline"),
    elapsed: document.getElementById("elapsed-time"),
    theme: document.getElementById("theme-toggle"),
    dialog: document.getElementById("media-dialog"),
    mediaPair: document.getElementById("media-pair"),
    mediaStatus: document.getElementById("media-status"),
    retry: document.getElementById("media-retry"),
    live: document.getElementById("tour-status"),
    heading: document.getElementById("stage-heading"),
  };

  const response = await fetch("static/data/explainer-evidence.json");
  if (!response.ok) throw new Error(`Evidence manifest failed with ${response.status}.`);
  const evidence = await response.json();
  const evidenceById = new Map(evidence.cases.map((item) => [item.id, item]));
  const media = createMediaController({ createVideo: browserVideoFactory, now: () => performance.now() });
  const clock = createRafClock();

  function updateChrome(scene, controllerState) {
    renderScene(elements.stage, elements.inspector, scene);
    renderChapterTabs(elements.tabs, scene);
    elements.timeline.value = String(scene.timeMs);
    elements.elapsed.textContent = `${formatTime(scene.timeMs)} / 01:30`;
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
    document.documentElement.dataset.theme = controllerState.theme;
    document.documentElement.dataset.stage = scene.chapter.id;
    elements.theme.innerHTML = controllerState.theme === "dark" ? '<span aria-hidden="true">&#9788;</span>' : '<span aria-hidden="true">&#9790;</span>';
    elements.theme.setAttribute("aria-label", controllerState.theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  const controller = createTourController({
    model: { deriveTourState },
    renderer: updateChrome,
    media,
    storage: window.localStorage,
    clock,
  });
  bindBeforeAfterRelease(document, controller);

  for (const control of [elements.play, elements.replay, elements.timeline, elements.theme]) control.disabled = false;

  elements.play.addEventListener("click", () => controller.getState().playing ? controller.pause() : controller.play());
  elements.previous.addEventListener("click", controller.previousChapter);
  elements.next.addEventListener("click", controller.nextChapter);
  elements.replay.addEventListener("click", controller.replayChapter);
  elements.timeline.addEventListener("input", (event) => controller.seek(Number(event.target.value)));
  elements.theme.addEventListener("click", () => controller.setTheme(controller.getState().theme === "dark" ? "light" : "dark"));
  elements.tabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-chapter-index]");
    if (!button) return;
    controller.seek(CHAPTERS[Number(button.dataset.chapterIndex)].startMs);
  });
  elements.stage.addEventListener("click", (event) => {
    const action = event.target.closest("[data-step]");
    if (action) controller.selectStep(Number(action.dataset.step));
  });
  elements.stage.addEventListener("keydown", (event) => {
    const action = event.target.closest?.("[data-step]");
    if (!action || !new Set(["ArrowLeft", "ArrowRight"]).has(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    controller.selectStep(clamp(Number(action.dataset.step) + offset, 0, 7));
  });
  elements.inspector.addEventListener("click", (event) => {
    const variant = event.target.closest("[data-variant]");
    if (variant) controller.setVariant(variant.dataset.variant);
  });
  elements.inspector.addEventListener("pointerdown", (event) => {
    if (event.target.closest("#before-after")) controller.setBefore(true);
  });
  for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
    elements.inspector.addEventListener(eventName, (event) => {
      if (event.target.closest?.("#before-after") || controller.getState().scene.before) controller.setBefore(false);
    });
  }
  elements.inspector.addEventListener("keydown", (event) => {
    if (!event.target.closest?.("#before-after") || !new Set(["Space", "Enter"]).has(event.code)) return;
    event.preventDefault();
    event.stopPropagation();
    controller.setBefore(true);
  });
  elements.inspector.addEventListener("keyup", (event) => {
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
    const loadPromise = media.loadCase(caseData, media.getState().generation + 1);
    mountMediaPair(elements.mediaPair, media.getVideos());
    const mediaState = await loadPromise;
    if (mediaState.status === "error") {
      elements.mediaStatus.textContent = "Recorded media is unavailable. The schematic tour remains active.";
      elements.retry.hidden = false;
    } else {
      elements.mediaStatus.textContent = `${caseData.note} ${caseData.playbackRate === 3 ? "Playback is set to 3x." : ""}`;
      media.sync(0, false);
    }
  }

  document.addEventListener("click", (event) => {
    const evidenceCase = event.target.closest?.("[data-evidence-case]");
    if (evidenceCase) openEvidence(evidenceCase.dataset.evidenceCase);
    if (event.target.closest?.("[data-dialog-close]")) elements.dialog.close();
  });
  elements.dialog.addEventListener("close", () => media.pause());
  elements.retry.addEventListener("click", async () => {
    elements.retry.hidden = true;
    elements.mediaStatus.textContent = "Retrying recorded media...";
    const retryPromise = media.retry();
    mountMediaPair(elements.mediaPair, media.getVideos());
    const result = await retryPromise;
    elements.retry.hidden = result.status !== "error";
    elements.mediaStatus.textContent = result.status === "ready" ? result.caseData.note : "Recorded media is still unavailable.";
  });
  document.addEventListener("visibilitychange", () => controller.onVisibilityChange(document.hidden));
  document.addEventListener("keydown", (event) => controller.handleKey(event));
  window.addEventListener("pagehide", () => { controller.dispose(); media.dispose(); }, { once: true });
}

if (typeof document !== "undefined") {
  initializeBrowserExplainer().catch((error) => {
    const status = document.getElementById("tour-status");
    if (status) status.textContent = `Explainer could not initialize: ${error.message}`;
  });
}
