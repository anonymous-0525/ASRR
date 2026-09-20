from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src", "poster"):
            value = attrs.get(key)
            if value:
                self.refs.append((tag, key, value))


def parse_page(path):
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


class StaticSiteTests(unittest.TestCase):
    def test_homepage_embeds_live_explainer_between_hero_and_abstract(self):
        html = (DOCS / "index.html").read_text(encoding="utf-8")
        hero_position = html.index('<header class="hero">')
        explainer_position = html.index('data-explainer-root="embedded"')
        abstract_position = html.index('id="abstract"')

        self.assertLess(hero_position, explainer_position)
        self.assertLess(explainer_position, abstract_position)
        self.assertNotIn('class="explainer-preview"', html)
        self.assertIn("two-minute guided explanation", html)
        self.assertIn('href="explainer.html"', html)

    def test_homepage_hero_uses_four_local_looping_video_tiles(self):
        html = (DOCS / "index.html").read_text(encoding="utf-8")
        video_tags = re.findall(r'<video\s+[^>]*class="hero__tile-video"[^>]*>', html)

        self.assertEqual(4, len(video_tags))
        for tag in video_tags:
            with self.subTest(tag=tag):
                self.assertIn(" muted", tag)
                self.assertIn(" loop", tag)
                self.assertIn(" playsinline", tag)
                self.assertRegex(tag, r'poster="static/images/[^"]+\.png"')
                self.assertRegex(tag, r'src="static/videos/[^"]+\.mp4"')

        self.assertEqual(2, html.count('class="hero__tile-role hero__tile-role--base"'))
        self.assertEqual(2, html.count('class="hero__tile-role hero__tile-role--asrr"'))

    def test_explainer_sources_remain_anonymous_and_tracking_free(self):
        source = "\n".join(
            (DOCS / name).read_text(encoding="utf-8")
            for name in ("index.html", "explainer.html")
        )
        self.assertNotRegex(source, r"NEBULIS|Google Analytics|gtag\(|plausible\.io")
        self.assertNotRegex(
            source,
            "clz" + "JY|Cheng" + "long|Fei" + "yang|Shuai" + "jun",
        )

        css = (DOCS / "static/css/explainer.css").read_text(encoding="utf-8")
        page_rules = re.findall(r"(?:html|body|\.explainer-app)\s*\{[^}]*\}", css, re.DOTALL)
        for rule in page_rules:
            fixed_minimums = [int(value) for value in re.findall(r"min-width:\s*(\d+)px", rule)]
            self.assertTrue(all(value <= 390 for value in fixed_minimums), rule)

    def test_explainer_shell_has_required_controls_and_local_resources(self):
        html_path = DOCS / "explainer.html"
        html = html_path.read_text(encoding="utf-8")
        parser = parse_page(html_path)

        required_ids = {
            "chapter-tabs", "stage", "inspector", "play-toggle",
            "previous-chapter", "next-chapter", "replay-chapter",
            "tour-timeline", "elapsed-time", "theme-toggle",
            "variant-toggle", "before-after", "media-dialog", "media-retry",
        }
        self.assertTrue(required_ids.issubset(parser.ids), required_ids - parser.ids)
        self.assertIn('data-explainer-root="standalone"', html)
        self.assertEqual(4, html.count('data-chapter-index="'))
        self.assertNotIn("<iframe", html)
        self.assertIn("two-minute interactive explanation", html)
        self.assertIn('href="static/css/explainer.css"', html)
        self.assertIn('type="module" src="static/js/explainer.js"', html)

        missing = []
        for _tag, _key, ref in parser.refs:
            if ref.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not (DOCS / ref).is_file():
                missing.append(ref)
        self.assertEqual([], missing)

    def test_explainer_evidence_manifest_is_complete(self):
        manifest_path = DOCS / "static/data/explainer-evidence.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual({"metrics", "cases"}, set(payload))
        self.assertEqual(
            {
                "pi05_mean_gain_pp": 14.4,
                "pi05_5k_success": [62.0, 80.9],
                "recorded_compute_reduction_percent": [25, 63],
                "real_robot_gain_pp": 6.7,
            },
            payload["metrics"],
        )
        self.assertEqual(
            {"pi05-libero10", "openvla-goal", "corn", "holder", "stack"},
            {case["id"] for case in payload["cases"]},
        )
        self.assertEqual(len(payload["cases"]), len({case["id"] for case in payload["cases"]}))

        case_keys = {
            "id", "policy", "task", "kind", "baseVideo", "asrrVideo",
            "basePoster", "asrrPoster", "playbackRate", "note", "authored",
        }
        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case_keys, set(case))
                self.assertIn(case["kind"], {"simulation", "real_robot"})
                self.assertGreater(case["playbackRate"], 0)
                self.assertIsInstance(case["authored"], bool)
                self.assertTrue(case["note"].strip())
                self.assertNotRegex(case["policy"], r"Octo|SmolVLA")
                for key in ("baseVideo", "asrrVideo", "basePoster", "asrrPoster"):
                    ref = case[key]
                    self.assertFalse(Path(ref).is_absolute(), (case["id"], key))
                    self.assertNotIn("..", Path(ref).parts, (case["id"], key))
                    self.assertTrue((DOCS / ref).is_file(), (case["id"], key))

        self.assertEqual(
            {"pi05-libero10", "corn"},
            {case["id"] for case in payload["cases"] if case["authored"]},
        )

    def test_local_page_references_exist(self):
        html_path = DOCS / "index.html"
        parser = PageParser()
        parser.feed(html_path.read_text(encoding="utf-8"))

        missing_files = []
        missing_anchors = []
        for _tag, _key, ref in parser.refs:
            if ref.startswith(("http://", "https://", "mailto:")):
                continue
            if ref.startswith("#"):
                if ref != "#" and ref[1:] not in parser.ids:
                    missing_anchors.append(ref)
                continue
            if not (DOCS / ref).exists():
                missing_files.append(ref)

        self.assertEqual([], missing_files)
        self.assertEqual([], missing_anchors)

    def test_public_files_do_not_expose_local_private_paths(self):
        patterns = [
            "/data/" + "private",
            r"\b" + "user" + r"7\b",
            r"\b" + "cheng" + r"long\b",
            "Cheng" + "long Zhang",
            "Fei" + "yang You",
            "Shuai" + "jun Liu",
            "Shu" + "yang Hao",
            "Cheng" + "yu Wu",
            "Ning" + "xin Su",
        ]
        matcher = re.compile("|".join(patterns), flags=re.IGNORECASE)
        checked_suffixes = {".html", ".css", ".js", ".md", ".py", ".toml", ".svg"}
        offenders = []

        for path in ROOT.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue
            if path.suffix not in checked_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if matcher.search(text):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual([], offenders)

    def test_expected_public_assets_are_present(self):
        expected = [
            DOCS / "index.html",
            DOCS / "static/css/site.css",
            DOCS / "static/js/site.js",
            DOCS / "static/files/asrr_paper.pdf",
            DOCS / "static/images/hero_rescue_refined_primary.png",
            DOCS / "static/images/hero_sim_primary.png",
            DOCS / "static/images/hero_sim_wrist.png",
            DOCS / "static/images/hero_real_corn.png",
            DOCS / "static/images/hero_real_holder.png",
            DOCS / "static/images/figure1_overview.png",
            DOCS / "static/images/figure2_method.png",
            DOCS / "static/images/figure3_act_horizon.png",
            DOCS / "static/images/figure4_checkpoint_context.png",
            DOCS / "static/images/table1_simulation_results.png",
            DOCS / "static/videos/vla_rescues/pi05_libero10_base.mp4",
            DOCS / "static/videos/vla_rescues/pi05_libero10_asrr.mp4",
            DOCS / "static/videos/vla_rescues/openvla_oft_goal_primary_base.mp4",
            DOCS / "static/videos/vla_rescues/openvla_oft_goal_primary_refined.mp4",
            DOCS / "static/videos/real_robot_tasks/corn_to_plate_base.mp4",
            DOCS / "static/videos/real_robot_tasks/corn_to_plate_asrr.mp4",
            DOCS / "static/videos/real_robot_tasks/block_to_holder_base.mp4",
            DOCS / "static/videos/real_robot_tasks/block_to_holder_asrr.mp4",
            DOCS / "static/videos/real_robot_tasks/block_stacking_base.mp4",
            DOCS / "static/videos/real_robot_tasks/block_stacking_asrr.mp4",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_video_scope_matches_final_paper(self):
        html = (DOCS / "index.html").read_text(encoding="utf-8")
        video_refs = sorted(set(re.findall(r'src="([^"]+\.mp4)"', html)))

        self.assertEqual(10, len(video_refs))
        self.assertNotIn("Octo", html)
        self.assertNotIn("SmolVLA", html)
        self.assertEqual(6, html.count('data-playback-rate="3"'))
        self.assertEqual(6, html.count('<span class="speed-badge">3×</span>'))

    def test_public_code_scope_matches_final_paper(self):
        expected = ["act", "diffusion_policy", "pi05", "openvla_oft"]
        for name in expected:
            self.assertTrue((ROOT / "examples" / name).is_dir(), name)
        self.assertFalse((ROOT / "examples" / "octo").exists())
        self.assertFalse((ROOT / "examples" / "smolvla").exists())


if __name__ == "__main__":
    unittest.main()
