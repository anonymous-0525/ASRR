import { mountExplainer } from "./explainer.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function applyPlaybackRate(video, attribute = "playbackRate") {
  const rate = Number(video.dataset[attribute] ?? 1);
  if (!Number.isFinite(rate) || rate <= 0) return;
  video.defaultPlaybackRate = rate;
  video.playbackRate = rate;
}

document.querySelectorAll("video[data-playback-rate]").forEach((video) => {
  const applyRate = () => applyPlaybackRate(video);
  video.addEventListener("loadedmetadata", applyRate);
  video.addEventListener("play", applyRate);
  applyRate();
});

const hero = document.querySelector(".hero");
const heroVideos = [...document.querySelectorAll(".hero__tile-video")];
let heroVisible = false;

function keepHeroVideoInSegment(video) {
  const start = Number(video.dataset.heroStartTime ?? 0);
  const end = Number(video.dataset.heroEndTime ?? video.duration);
  if (video.readyState < 1 || !Number.isFinite(start) || !Number.isFinite(end)) return;
  if (video.currentTime < start || video.currentTime >= end) video.currentTime = start;
}

function syncHeroVideos() {
  const shouldPlay = heroVisible && !document.hidden && !reducedMotion.matches;
  for (const video of heroVideos) {
    applyPlaybackRate(video, "heroPlaybackRate");
    keepHeroVideoInSegment(video);
    if (shouldPlay) {
      video.play().catch(() => {});
    } else {
      video.pause();
      if (reducedMotion.matches) video.currentTime = 0;
    }
  }
}

for (const video of heroVideos) {
  video.addEventListener("loadedmetadata", syncHeroVideos);
  video.addEventListener("play", () => applyPlaybackRate(video, "heroPlaybackRate"));
  video.addEventListener("timeupdate", () => keepHeroVideoInSegment(video));
}

if (hero && "IntersectionObserver" in window) {
  const heroObserver = new IntersectionObserver(([entry]) => {
    heroVisible = entry.isIntersecting && entry.intersectionRatio >= 0.2;
    syncHeroVideos();
  }, { threshold: [0, 0.2, 0.6] });
  heroObserver.observe(hero);
} else {
  heroVisible = true;
}

document.addEventListener("visibilitychange", syncHeroVideos);
reducedMotion.addEventListener?.("change", syncHeroVideos);
syncHeroVideos();

const embeddedExplainer = document.querySelector('[data-explainer-root="embedded"]');
if (embeddedExplainer) {
  mountExplainer(embeddedExplainer, {
    mode: "embedded",
    autoplayWhenVisible: !reducedMotion.matches,
  }).catch((error) => {
    const status = embeddedExplainer.querySelector("#tour-status");
    if (status) status.textContent = `Explainer could not initialize: ${error.message}`;
  });
}
