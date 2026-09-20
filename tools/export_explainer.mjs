import { spawn } from "node:child_process";
import { mkdir, writeFile, rm } from "node:fs/promises";
import { resolve, join } from "node:path";
import { parseArgs } from "node:util";
import { CHAPTERS, TOUR_DURATION_MS } from "../docs/static/js/explainer-model.mjs";

const { values } = parseArgs({ options: {
  session: { type: "string" },
  output: { type: "string" },
  display: { type: "string", default: ":97" },
  webdriver: { type: "string", default: "http://127.0.0.1:4444" },
  ffmpeg: { type: "string", default: "ffmpeg" },
  ffprobe: { type: "string", default: "ffprobe" },
} });
if (!values.session || !values.output) {
  throw new Error("Required: --session WEBDRIVER_SESSION --output DIRECTORY. Open explainer.html?export=1 in a 1920x1080 kiosk browser first.");
}
const output = resolve(values.output);
await mkdir(output, { recursive: true });
const endpoint = `${values.webdriver}/session/${values.session}`;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function command(path, body) {
  const response = await fetch(endpoint + path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const { value } = await response.json();
  if (value?.error) throw new Error(JSON.stringify(value));
  return value;
}
const execute = (script, args = []) => command("/execute/sync", { script, args });
async function seek(timeMs) {
  await execute('const t=document.querySelector("#tour-timeline");t.value=arguments[0];t.dispatchEvent(new Event("input",{bubbles:true}));', [timeMs]);
}
async function waitFor(script, timeoutMs = 20_000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    if (await execute(script)) return;
    await sleep(100);
  }
  throw new Error(`Browser did not become ready: ${script}`);
}
function run(program, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(program, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", code => code === 0 ? resolve(stdout) : reject(new Error(stderr)));
  });
}

const preflight = await execute(`return {
  size:[innerWidth,innerHeight], clean:document.body.hasAttribute('data-export'),
  controls:[...document.querySelectorAll('button,input,a')].filter(e=>e.getClientRects().length).length
};`);
if (!preflight.clean || preflight.controls || preflight.size.join("x") !== "1920x1080") {
  throw new Error(`Capture preflight failed: ${JSON.stringify(preflight)}`);
}

const frames = [[12_500,"local-error"], [40_000,"editable-interface"], [58_000,"action-only"],
  [70_500,"context-conditioned"], [80_000,"simulation"], [97_000,"real-robot"]];
for (const [time, name] of frames) {
  await seek(time);
  if (time >= CHAPTERS[3].startMs) {
    await waitFor('return document.querySelectorAll(".authored-media-pair video").length===2 && [...document.querySelectorAll(".authored-media-pair video")].every(v=>v.readyState>=2&&!v.seeking&&!v.controls)');
  }
  await sleep(300);
  await writeFile(join(output, `${name}.png`), Buffer.from(await command("/screenshot"), "base64"));
}
await seek(0);
await sleep(500);

const capturePath = join(output, "capture-source.mp4");
const duration = TOUR_DURATION_MS / 1000;
const ffmpeg = spawn(values.ffmpeg, ["-hide_banner", "-y", "-f", "x11grab", "-video_size", "1920x1080",
  "-framerate", "30", "-draw_mouse", "0", "-i", values.display,
  "-t", String(duration + 5), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
  "-threads", "8", "-pix_fmt", "yuv420p", "-movflags", "+faststart", capturePath],
  { stdio: ["ignore", "ignore", "pipe"] });
let log = "", captureEpoch = null, captureDone = false;
const completion = new Promise((resolve, reject) => {
  ffmpeg.on("error", reject);
  ffmpeg.on("close", code => {
    captureDone = true;
    code === 0 ? resolve() : reject(new Error(log));
  });
});
completion.catch(() => {});
ffmpeg.stderr.on("data", chunk => {
  log += chunk;
  const match = log.match(/start: (\d+\.\d+)/);
  if (match) captureEpoch = Number(match[1]);
});
try {
  const readyDeadline = Date.now() + 15_000;
  while (captureEpoch === null && !captureDone && Date.now() < readyDeadline) await sleep(50);
  if (captureEpoch === null) throw new Error(`No X11 capture timestamp: ${log}`);
  // Align the video to the browser clock using the first X11 frame timestamp.
  const playbackEpoch = await execute(`document.querySelector('#play-toggle').click();
    return (performance.timeOrigin+performance.now())/1000;`);
  const trimStart = playbackEpoch - captureEpoch;
  if (trimStart < 0 || trimStart > 4) throw new Error(`Invalid capture pre-roll: ${trimStart}s`);
  console.log(`Recording ${duration}s at 1080p/30fps; pre-roll ${trimStart.toFixed(3)}s.`);
  let lastChapter = null;
  while (!captureDone) {
    await sleep(1000);
    const state = await execute(`return {time:Number(document.querySelector('#tour-timeline').value),
      chapter:document.querySelector('#stage-heading').textContent,
      videos:[...document.querySelectorAll('.authored-media-pair video')].map(v=>({rate:v.playbackRate,time:v.currentTime,ready:v.readyState,controls:v.controls}))};`);
    if (state.chapter !== lastChapter) { console.log(JSON.stringify(state)); lastChapter = state.chapter; }
    if (state.time >= TOUR_DURATION_MS) break;
  }
  await completion;
  await writeFile(join(output, "capture.log"), log);
  const master = join(output, "ASRR_explainer_clean_1080p.mp4");
  const encode = ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "8",
    "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart"];
  await run(values.ffmpeg, ["-hide_banner", "-loglevel", "error", "-y", "-ss", String(trimStart),
    "-i", capturePath, "-t", String(duration), ...encode, master]);
  for (const chapter of CHAPTERS) {
    const filename = `${String(chapter.index + 1).padStart(2,"0")}_${chapter.id}.mp4`;
    await run(values.ffmpeg, ["-hide_banner", "-loglevel", "error", "-y", "-ss", String(chapter.startMs/1000),
      "-i", master, "-t", String((chapter.endMs-chapter.startMs)/1000), ...encode, join(output, filename)]);
  }
  const probe = JSON.parse(await run(values.ffprobe, ["-v", "error", "-show_entries",
    "stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames:format=duration,size", "-of", "json", master]));
  const stream = probe.streams.find(s=>s.codec_type === "video");
  if (stream?.width !== 1920 || stream.height !== 1080 || stream.r_frame_rate !== "30/1" ||
      Math.abs(Number(probe.format.duration)-duration)>0.05 || probe.streams.length !== 1) {
    throw new Error(`Unexpected export: ${JSON.stringify(probe)}`);
  }
  await writeFile(join(output, "export.json"), JSON.stringify({
    durationSeconds:duration, trimStartSeconds:trimStart, format:probe,
    chapters:CHAPTERS, audio:"none", realRobotPlaybackRate:3,
  }, null, 2));
  await rm(capturePath);
  console.log(`Export verified: ${master}`);
} finally {
  if (!captureDone) ffmpeg.kill("SIGTERM");
}
