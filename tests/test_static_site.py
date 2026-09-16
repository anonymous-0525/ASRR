from html.parser import HTMLParser
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


class StaticSiteTests(unittest.TestCase):
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
            DOCS / "static/images/figure1_overview.png",
            DOCS / "static/images/figure2_method.png",
            DOCS / "static/images/figure3_act_horizon.png",
            DOCS / "static/images/figure4_checkpoint_context.png",
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
        self.assertEqual(6, html.count('data-playback-rate="2"'))
        self.assertEqual(6, html.count('<span class="speed-badge">2×</span>'))


if __name__ == "__main__":
    unittest.main()
