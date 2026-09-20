import test from "node:test";
import assert from "node:assert/strict";

import { createMediaController } from "../docs/static/js/explainer-media.mjs";

const CASE_A = {
  id: "corn",
  policy: "Piper pi0.5",
  task: "Corn to plate",
  kind: "real_robot",
  baseVideo: "base.mp4",
  asrrVideo: "asrr.mp4",
  basePoster: "base.png",
  asrrPoster: "asrr.png",
  playbackRate: 3,
  note: "Recorded physical execution.",
};

const CASE_B = {
  ...CASE_A,
  id: "pi05-libero10",
  policy: "pi0.5",
  task: "LIBERO-10 rescue",
  kind: "simulation",
  playbackRate: 1,
};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

function makeVideo(loadPromise = Promise.resolve(), resetRateOnLoad = false) {
  return {
    currentTime: 0,
    playbackRate: 1,
    muted: false,
    playsInline: false,
    preload: "",
    paused: true,
    playCalls: 0,
    pauseCalls: 0,
    load() {
      if (resetRateOnLoad) this.playbackRate = 1;
      return loadPromise;
    },
    play() { this.paused = false; this.playCalls += 1; return Promise.resolve(); },
    pause() { this.paused = true; this.pauseCalls += 1; },
    removeAttribute() {},
  };
}

test("stale media completion cannot restart playback", async () => {
  const gate = deferred();
  const videos = [];
  const media = createMediaController({
    createVideo: () => {
      const video = makeVideo(gate.promise);
      videos.push(video);
      return video;
    },
    now: () => 0,
  });

  const first = media.loadCase(CASE_A, 1);
  media.dispose();
  gate.resolve();
  await first;

  assert.equal(videos.reduce((sum, video) => sum + video.playCalls, 0), 0);
  assert.equal(media.getState().status, "idle");
});

test("real robot media retains declared 3x playback", async () => {
  const videos = [];
  const media = createMediaController({
    createVideo: () => {
      const video = makeVideo();
      videos.push(video);
      return video;
    },
    now: () => 0,
  });

  assert.equal(typeof media.loadPair, "function");
  await media.loadPair(CASE_A, 1);
  media.sync(4, true);

  assert.equal(videos.length, 2);
  assert.ok(videos.every((video) => video.playbackRate === 3));
  assert.ok(videos.every((video) => video.currentTime === 12));
  assert.ok(videos.every((video) => video.playCalls === 1));
});

test("switching an authored pair pauses and releases the stale videos", async () => {
  const videos = [];
  const media = createMediaController({
    createVideo: () => {
      const video = makeVideo();
      videos.push(video);
      return video;
    },
    now: () => 0,
  });

  await media.loadPair(CASE_B, 1);
  const stale = media.getVideos();
  await media.loadPair(CASE_A, 2);
  media.sync(2, false);

  assert.ok(stale.every((video) => video.pauseCalls >= 1));
  assert.ok(media.getVideos().every((video) => video.playbackRate === 3));
  assert.ok(media.getVideos().every((video) => video.currentTime === 6));
});

test("metadata loading cannot reset the declared playback rate", async () => {
  const videos = [];
  const media = createMediaController({
    createVideo: () => {
      const video = makeVideo(Promise.resolve(), true);
      videos.push(video);
      return video;
    },
    now: () => 0,
  });

  await media.loadCase(CASE_A, 1);

  assert.ok(videos.every((video) => video.playbackRate === 3));
});

test("pause freezes both videos", async () => {
  const videos = [];
  const media = createMediaController({
    createVideo: () => {
      const video = makeVideo();
      videos.push(video);
      return video;
    },
    now: () => 0,
  });
  await media.loadCase(CASE_A, 1);
  media.sync(2.5, true);
  media.pause();

  assert.ok(videos.every((video) => video.pauseCalls === 1));
  assert.equal(media.getState().playing, false);
});

test("failed load exposes retry and retry clears the error", async () => {
  let batch = 0;
  const media = createMediaController({
    createVideo: () => {
      const currentBatch = Math.floor(batch / 2);
      batch += 1;
      return makeVideo(currentBatch === 0 ? Promise.reject(new Error("offline")) : Promise.resolve());
    },
    now: () => 0,
  });

  await media.loadCase(CASE_A, 1);
  assert.equal(media.getState().status, "error");
  assert.match(media.getState().error.message, /offline/);

  await media.retry();
  assert.equal(media.getState().status, "ready");
  assert.equal(media.getState().error, null);
});

test("invalid evidence is rejected before video creation", async () => {
  let createCalls = 0;
  const media = createMediaController({
    createVideo: () => { createCalls += 1; return makeVideo(); },
    now: () => 0,
  });

  await assert.rejects(() => media.loadCase({ ...CASE_A, baseVideo: "" }, 1), /baseVideo/);
  assert.equal(createCalls, 0);
});
