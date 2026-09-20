const REQUIRED_CASE_FIELDS = [
  "id",
  "policy",
  "task",
  "kind",
  "baseVideo",
  "asrrVideo",
  "basePoster",
  "asrrPoster",
  "playbackRate",
  "note",
];

function validateCase(caseData) {
  if (!caseData || typeof caseData !== "object") {
    throw new TypeError("Evidence case must be an object.");
  }
  for (const field of REQUIRED_CASE_FIELDS) {
    if (caseData[field] === undefined || caseData[field] === null || caseData[field] === "") {
      throw new TypeError(`Evidence case requires ${field}.`);
    }
  }
  if (!Number.isFinite(caseData.playbackRate) || caseData.playbackRate <= 0) {
    throw new RangeError("Evidence playbackRate must be positive.");
  }
  if (!new Set(["simulation", "real_robot"]).has(caseData.kind)) {
    throw new RangeError("Evidence kind must be simulation or real_robot.");
  }
}

function configureVideo(video, playbackRate) {
  video.preload = "metadata";
  video.muted = true;
  video.playsInline = true;
  video.playbackRate = playbackRate;
}

async function loadVideo(video) {
  const result = video.load();
  if (result && typeof result.then === "function") {
    await result;
    return;
  }
  if (video.readyState >= 1 || typeof video.addEventListener !== "function") return;

  await new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener("loadedmetadata", done);
      video.removeEventListener("error", failed);
      resolve();
    };
    const failed = () => {
      video.removeEventListener("loadedmetadata", done);
      video.removeEventListener("error", failed);
      reject(video.error ?? new Error("Media metadata failed to load."));
    };
    video.addEventListener("loadedmetadata", done, { once: true });
    video.addEventListener("error", failed, { once: true });
  });
}

export function createMediaController({ createVideo, now = () => Date.now() }) {
  if (typeof createVideo !== "function") {
    throw new TypeError("createMediaController requires createVideo.");
  }

  let generation = 0;
  let videos = [];
  let currentCase = null;
  let status = "idle";
  let error = null;
  let playing = false;
  let requestedAt = null;

  function stopVideos(release = false) {
    for (const video of videos) {
      video.pause?.();
      if (release) {
        video.removeAttribute?.("src");
        video.load?.();
      }
    }
    playing = false;
  }

  async function loadCase(caseData, token) {
    validateCase(caseData);
    const localGeneration = Math.max(generation + 1, Number.isFinite(token) ? token : 0);
    generation = localGeneration;
    stopVideos(true);
    currentCase = caseData;
    status = "loading";
    error = null;
    requestedAt = now();

    videos = [
      createVideo({ role: "base", src: caseData.baseVideo, poster: caseData.basePoster }),
      createVideo({ role: "asrr", src: caseData.asrrVideo, poster: caseData.asrrPoster }),
    ];
    for (const video of videos) configureVideo(video, caseData.playbackRate);

    try {
      await Promise.all(videos.map(loadVideo));
      if (localGeneration !== generation) return getState();
      status = "ready";
    } catch (loadError) {
      if (localGeneration !== generation) return getState();
      status = "error";
      error = loadError instanceof Error ? loadError : new Error(String(loadError));
      stopVideos(false);
    }
    return getState();
  }

  function sync(timeSeconds, shouldPlay) {
    if (status !== "ready") return;
    const safeTime = Math.max(0, Number.isFinite(timeSeconds) ? timeSeconds : 0);
    const localGeneration = generation;
    for (const video of videos) {
      if (Math.abs((video.currentTime ?? 0) - safeTime) > 0.12) video.currentTime = safeTime;
      video.playbackRate = currentCase.playbackRate;
      if (shouldPlay) {
        Promise.resolve(video.play?.()).catch(() => {
          if (localGeneration === generation) {
            status = "error";
            error = new Error("Recorded media playback was blocked.");
          }
        });
      } else {
        video.pause?.();
      }
    }
    playing = Boolean(shouldPlay);
  }

  function pause() {
    stopVideos(false);
  }

  async function retry() {
    if (!currentCase) return getState();
    return loadCase(currentCase, generation + 1);
  }

  function dispose() {
    generation += 1;
    stopVideos(true);
    videos = [];
    currentCase = null;
    status = "idle";
    error = null;
    requestedAt = null;
  }

  function getState() {
    return {
      generation,
      status,
      error,
      playing,
      caseData: currentCase,
      requestedAt,
    };
  }

  function getVideos() {
    return [...videos];
  }

  return { loadCase, sync, pause, retry, dispose, getState, getVideos };
}
