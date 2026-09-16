document.querySelectorAll("video[data-playback-rate]").forEach((video) => {
  const rate = Number(video.dataset.playbackRate);
  if (!Number.isFinite(rate) || rate <= 0) {
    return;
  }

  const applyRate = () => {
    video.defaultPlaybackRate = rate;
    video.playbackRate = rate;
  };

  video.addEventListener("loadedmetadata", applyRate, { once: true });
  video.addEventListener("play", applyRate);
});
