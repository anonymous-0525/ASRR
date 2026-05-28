const VIDEO_ROOT = "static/videos/vla_rescues";

const SUITE_COLUMNS = [
  ["Spatial", "Base"],
  ["Spatial", "ASRR"],
  ["Object", "Base"],
  ["Object", "ASRR"],
  ["Goal", "Base"],
  ["Goal", "ASRR"],
  ["LIBERO-10", "Base"],
  ["LIBERO-10", "ASRR"],
];

const RESCUE_ROWS = [
  {
    method: "OpenVLA-OFT",
    camera: "Primary camera",
    cells: [
      ["openvla_oft_spatial_primary_base.mp4", "Base failure", "Spatial: black bowl to the plate."],
      ["openvla_oft_spatial_primary_refined.mp4", "Refined rescue", "ASRR completes the same seed."],
      ["openvla_oft_object_primary_base.mp4", "Base failure", "Object: cream cheese to the basket."],
      ["openvla_oft_object_primary_refined.mp4", "Refined rescue", "ASRR completes the object transfer."],
      ["openvla_oft_goal_primary_base.mp4", "Base failure", "Goal: wine bottle on the cabinet."],
      ["openvla_oft_goal_primary_refined.mp4", "Refined rescue", "ASRR completes the placement."],
      ["openvla_oft_libero10_primary_base.mp4", "Base failure", "LIBERO-10: black bowl and drawer."],
      ["openvla_oft_libero10_primary_refined.mp4", "Refined rescue", "ASRR recovers the sequence."],
    ],
  },
  {
    method: "OpenVLA-OFT",
    camera: "Wrist camera",
    cells: [
      ["openvla_oft_spatial_wrist_base.mp4", "Base failure", "Spatial wrist stream."],
      ["openvla_oft_spatial_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["openvla_oft_object_wrist_base.mp4", "Base failure", "Object wrist stream."],
      ["openvla_oft_object_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["openvla_oft_goal_wrist_base.mp4", "Base failure", "Goal wrist stream."],
      ["openvla_oft_goal_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["openvla_oft_libero10_wrist_base.mp4", "Base failure", "LIBERO-10 wrist stream."],
      ["openvla_oft_libero10_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
    ],
  },
  {
    method: "Octo",
    camera: "Primary camera",
    cells: [
      ["octo_spatial_primary_base_rot180.mp4", "Base failure", "Spatial: black bowl from drawer to plate."],
      ["octo_spatial_primary_refined_rot180.mp4", "Refined rescue", "ASRR completes the same state."],
      ["octo_object_primary_base_rot180.mp4", "Base failure", "Object: salad dressing to basket."],
      ["octo_object_primary_refined_rot180.mp4", "Refined rescue", "ASRR completes the transfer."],
      ["octo_goal_primary_base_rot180.mp4", "Base failure", "Goal: plate movement task."],
      ["octo_goal_primary_refined_rot180.mp4", "Refined rescue", "ASRR completes the movement."],
      ["octo_libero10_primary_base_rot180.mp4", "Base failure", "LIBERO-10 selected sequence."],
      ["octo_libero10_primary_refined_rot180.mp4", "Refined rescue", "ASRR recovers the sequence."],
    ],
  },
  {
    method: "Octo",
    camera: "Wrist camera",
    cells: [
      ["octo_spatial_wrist_base_rot180.mp4", "Base failure", "Spatial wrist stream."],
      ["octo_spatial_wrist_refined_rot180.mp4", "Refined rescue", "Recovered wrist stream."],
      ["octo_object_wrist_base_rot180.mp4", "Base failure", "Object wrist stream."],
      ["octo_object_wrist_refined_rot180.mp4", "Refined rescue", "Recovered wrist stream."],
      ["octo_goal_wrist_base_rot180.mp4", "Base failure", "Goal wrist stream."],
      ["octo_goal_wrist_refined_rot180.mp4", "Refined rescue", "Recovered wrist stream."],
      ["octo_libero10_wrist_base_rot180.mp4", "Base failure", "LIBERO-10 wrist stream."],
      ["octo_libero10_wrist_refined_rot180.mp4", "Refined rescue", "Recovered wrist stream."],
    ],
  },
  {
    method: "pi0.5",
    camera: "Primary camera",
    cells: [
      ["pi05_spatial_primary_base.mp4", "Base failure", "Spatial selected case."],
      ["pi05_spatial_primary_refined.mp4", "Refined rescue", "Adapter rescue on the same case."],
      ["pi05_object_primary_base.mp4", "Base failure", "Object selected case."],
      ["pi05_object_primary_refined.mp4", "Refined rescue", "Adapter rescue on the same case."],
      ["pi05_goal_primary_base.mp4", "Base failure", "Goal selected case."],
      ["pi05_goal_primary_refined.mp4", "Refined rescue", "Adapter rescue on the same case."],
      ["pi05_libero10_primary_base.mp4", "Base failure", "LIBERO-10 selected case."],
      ["pi05_libero10_primary_refined.mp4", "Refined rescue", "Adapter rescue on the same case."],
    ],
  },
  {
    method: "pi0.5",
    camera: "Wrist camera",
    cells: [
      ["pi05_spatial_wrist_base.mp4", "Base failure", "Spatial wrist stream."],
      ["pi05_spatial_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["pi05_object_wrist_base.mp4", "Base failure", "Object wrist stream."],
      ["pi05_object_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["pi05_goal_wrist_base.mp4", "Base failure", "Goal wrist stream."],
      ["pi05_goal_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["pi05_libero10_wrist_base.mp4", "Base failure", "LIBERO-10 wrist stream."],
      ["pi05_libero10_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
    ],
  },
  {
    method: "SmolVLA",
    camera: "Primary camera",
    cells: [
      ["smolvla_spatial_primary_base.mp4", "Base failure", "Spatial: black bowl to the plate."],
      ["smolvla_spatial_primary_refined.mp4", "Refined rescue", "ASRR completes the placement."],
      ["smolvla_object_primary_base.mp4", "Base failure", "Object: alphabet soup to basket."],
      ["smolvla_object_primary_refined.mp4", "Refined rescue", "ASRR completes the transfer."],
      ["smolvla_goal_primary_base.mp4", "Base failure", "Goal: open the middle drawer."],
      ["smolvla_goal_primary_refined.mp4", "Refined rescue", "ASRR opens the drawer."],
      ["smolvla_libero10_primary_base.mp4", "Base failure", "LIBERO-10: moka pot on the stove."],
      ["smolvla_libero10_primary_refined.mp4", "Refined rescue", "ASRR completes the stove task."],
    ],
  },
  {
    method: "SmolVLA",
    camera: "Wrist camera",
    cells: [
      ["smolvla_spatial_wrist_base.mp4", "Base failure", "Spatial wrist stream."],
      ["smolvla_spatial_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["smolvla_object_wrist_base.mp4", "Base failure", "Object wrist stream."],
      ["smolvla_object_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["smolvla_goal_wrist_base.mp4", "Base failure", "Goal wrist stream."],
      ["smolvla_goal_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
      ["smolvla_libero10_wrist_base.mp4", "Base failure", "LIBERO-10 wrist stream."],
      ["smolvla_libero10_wrist_refined.mp4", "Refined rescue", "Recovered wrist stream."],
    ],
  },
];

function appendTextElement(parent, tagName, text, className) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

function createVideoCard([fileName, title, caption]) {
  const card = document.createElement("article");
  card.className = fileName ? "video-card" : "video-card video-card--pending";

  if (fileName) {
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";

    const source = document.createElement("source");
    source.src = `${VIDEO_ROOT}/${fileName}`;
    source.type = "video/mp4";
    video.appendChild(source);
    card.appendChild(video);
  }

  appendTextElement(card, "strong", title);
  appendTextElement(card, "p", caption);
  return card;
}

function renderRescueBoard() {
  const board = document.getElementById("rescue-video-board");
  if (!board) {
    return;
  }

  board.replaceChildren();
  appendTextElement(board, "div", "Method / camera", "rescue-board__corner");

  SUITE_COLUMNS.forEach(([suite, label]) => {
    const head = appendTextElement(board, "div", suite, "rescue-board__head");
    appendTextElement(head, "span", label);
  });

  RESCUE_ROWS.forEach((row) => {
    const rowHead = appendTextElement(board, "div", row.method, "rescue-board__rowhead");
    appendTextElement(rowHead, "span", row.camera);
    row.cells.forEach((cell) => board.appendChild(createVideoCard(cell)));
  });
}

renderRescueBoard();
