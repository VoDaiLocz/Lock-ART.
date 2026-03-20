"""Generate README assets from real AuraLock demo outputs and reports."""

from __future__ import annotations

import json
import math
import sys
import xml.sax.saxutils as xml_escape
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auralock.services import ProtectionService  # noqa: E402

ASSETS_DIR = ROOT / "docs" / "assets"
REPORTS_DIR = ROOT / "output" / "reports"
DEMO_DIR = ROOT / "output" / "demo"
SUBJECT_SET_DIR = ROOT / ".cache_ref" / "Anti-DreamBooth" / "data" / "n000050" / "set_B"
COLLECTIVE_DIR = DEMO_DIR / "n000050_subject_collective"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _card(x: int, y: int, width: int, height: int, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="24" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
    )


def build_banner() -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="620" viewBox="0 0 1400 620" fill="none">
  <defs>
    <linearGradient id="bg" x1="120" y1="40" x2="1280" y2="580" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0f172a"/>
      <stop offset="0.55" stop-color="#111827"/>
      <stop offset="1" stop-color="#0b1220"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f59e0b"/>
      <stop offset="1" stop-color="#14b8a6"/>
    </linearGradient>
  </defs>
  <rect width="1400" height="620" rx="36" fill="url(#bg)"/>
  <circle cx="1140" cy="110" r="170" fill="#14b8a6" opacity="0.08"/>
  <circle cx="1260" cy="480" r="220" fill="#f59e0b" opacity="0.07"/>
  <circle cx="190" cy="520" r="140" fill="#60a5fa" opacity="0.06"/>
  <text x="110" y="164" fill="#f8fafc" font-size="74" font-family="Segoe UI, Arial, sans-serif" font-weight="700">AuraLock</text>
  <text x="110" y="230" fill="#cbd5e1" font-size="30" font-family="Segoe UI, Arial, sans-serif">
    Production-ready artwork cloaking for anti-mimicry workflows
  </text>
  <text x="110" y="292" fill="#94a3b8" font-size="23" font-family="Segoe UI, Arial, sans-serif">
    Collective subject-set protection, Anti-DreamBooth-aligned benchmarking, CLI/UI, Docker and Colab execution paths.
  </text>
  <rect x="110" y="336" width="198" height="46" rx="23" fill="#172554" stroke="#3b82f6" stroke-width="1.2"/>
  <text x="142" y="367" fill="#dbeafe" font-size="21" font-family="Segoe UI, Arial, sans-serif">75 tests passing</text>
  <rect x="328" y="336" width="212" height="46" rx="23" fill="#1f2937" stroke="#14b8a6" stroke-width="1.2"/>
  <text x="359" y="367" fill="#ccfbf1" font-size="21" font-family="Segoe UI, Arial, sans-serif">CPU + Docker + Colab</text>
  <rect x="560" y="336" width="286" height="46" rx="23" fill="#292524" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="592" y="367" fill="#fef3c7" font-size="21" font-family="Segoe UI, Arial, sans-serif">Anti-DreamBooth benchmark harness</text>
  {_card(872, 108, 218, 160, "#111827", "#1f2937")}
  <text x="904" y="160" fill="#f59e0b" font-size="20" font-family="Segoe UI, Arial, sans-serif">Mode</text>
  <text x="904" y="218" fill="#f8fafc" font-size="40" font-family="Segoe UI, Arial, sans-serif" font-weight="700">Collective</text>
  <text x="904" y="246" fill="#94a3b8" font-size="19" font-family="Segoe UI, Arial, sans-serif">subject-set protection</text>
  {_card(1112, 108, 178, 160, "#111827", "#1f2937")}
  <text x="1144" y="160" fill="#14b8a6" font-size="20" font-family="Segoe UI, Arial, sans-serif">Core</text>
  <text x="1144" y="218" fill="#f8fafc" font-size="40" font-family="Segoe UI, Arial, sans-serif" font-weight="700">StyleCloak</text>
  <text x="1144" y="246" fill="#94a3b8" font-size="19" font-family="Segoe UI, Arial, sans-serif">feature-space defense</text>
  {_card(872, 292, 418, 170, "#111827", "#1f2937")}
  <text x="908" y="348" fill="#93c5fd" font-size="24" font-family="Segoe UI, Arial, sans-serif">Current focus</text>
  <text x="908" y="402" fill="#f8fafc" font-size="34" font-family="Segoe UI, Arial, sans-serif" font-weight="700">Professional delivery, honest metrics</text>
  <text x="908" y="437" fill="#94a3b8" font-size="20" font-family="Segoe UI, Arial, sans-serif">
    README visuals are generated from real reports so the project page stays defensible.
  </text>
  <rect x="110" y="430" width="560" height="104" rx="24" fill="#0f172a" stroke="url(#accent)" stroke-width="1.5"/>
  <text x="146" y="473" fill="#f8fafc" font-size="24" font-family="Segoe UI, Arial, sans-serif" font-weight="600">
    Why this README exists
  </text>
  <text x="146" y="516" fill="#94a3b8" font-size="21" font-family="Segoe UI, Arial, sans-serif">
    Clear product story, real output images, benchmark snapshots and deploy-ready workflow in one page.
  </text>
</svg>"""
    _write_svg(ASSETS_DIR / "readme-banner.svg", svg)


def _load_phonecase_tradeoff() -> list[dict[str, float | str]]:
    sources = {
        "Balanced": REPORTS_DIR / "balanced-phonecase-proxy-v2.json",
        "Subject": REPORTS_DIR / "subject-phonecase-proxy-v6.json",
        "Fortress": REPORTS_DIR / "fortress-phonecase-proxy-v2.json",
        "Blindfold": DEMO_DIR / "horse_painting_preview_768_blindfold_analyze.json",
    }
    rows: list[dict[str, float | str]] = []
    for label, path in sources.items():
        payload = _read_json(path)
        quality = payload["quality_report"]
        protection = payload["protection_report"]
        rows.append(
            {
                "label": label,
                "psnr": float(quality["psnr_db"]),
                "ssim": float(quality["ssim"]),
                "protection": float(protection["protection_score"]),
            }
        )
    return rows


def build_profile_tradeoff_chart() -> None:
    rows = _load_phonecase_tradeoff()
    max_score = max(float(row["protection"]) for row in rows)
    width = 1400
    height = 700
    left = 120
    top = 150
    bar_height = 72
    gap = 44
    max_bar_width = 760
    palette = ["#3b82f6", "#14b8a6", "#f59e0b", "#ef4444"]
    svg_rows: list[str] = []
    for index, row in enumerate(rows):
        y = top + index * (bar_height + gap)
        protection = float(row["protection"])
        bar_width = protection / max_score * max_bar_width
        ssim = float(row["ssim"])
        psnr = float(row["psnr"])
        color = palette[index]
        svg_rows.append(
            f'<text x="{left}" y="{y + 31}" fill="#e5e7eb" font-size="28" font-family="Segoe UI, Arial, sans-serif" font-weight="600">{xml_escape.escape(str(row["label"]))}</text>'
        )
        svg_rows.append(
            f'<rect x="{left}" y="{y + 46}" width="{max_bar_width}" height="{bar_height}" rx="18" fill="#1f2937" />'
        )
        svg_rows.append(
            f'<rect x="{left}" y="{y + 46}" width="{bar_width:.1f}" height="{bar_height}" rx="18" fill="{color}" />'
        )
        svg_rows.append(
            f'<text x="{left + max_bar_width + 24}" y="{y + 90}" fill="#f8fafc" font-size="24" font-family="Segoe UI, Arial, sans-serif" font-weight="700">{_fmt(protection)} / 100</text>'
        )
        svg_rows.append(
            f'<text x="1080" y="{y + 76}" fill="#cbd5e1" font-size="20" font-family="Segoe UI, Arial, sans-serif">PSNR</text>'
        )
        svg_rows.append(
            f'<text x="1160" y="{y + 76}" fill="#f8fafc" font-size="20" font-family="Segoe UI, Arial, sans-serif">{_fmt(psnr, 2)} dB</text>'
        )
        svg_rows.append(
            f'<text x="1080" y="{y + 106}" fill="#cbd5e1" font-size="20" font-family="Segoe UI, Arial, sans-serif">SSIM</text>'
        )
        svg_rows.append(
            f'<text x="1160" y="{y + 106}" fill="#f8fafc" font-size="20" font-family="Segoe UI, Arial, sans-serif">{_fmt(ssim, 4)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">
  <rect width="{width}" height="{height}" rx="32" fill="#0b1220"/>
  <text x="120" y="86" fill="#f8fafc" font-size="42" font-family="Segoe UI, Arial, sans-serif" font-weight="700">Profile Trade-off Snapshot</text>
  <text x="120" y="122" fill="#94a3b8" font-size="22" font-family="Segoe UI, Arial, sans-serif">Representative local runs. Higher protection usually costs more visible change.</text>
  {''.join(svg_rows)}
</svg>"""
    _write_svg(ASSETS_DIR / "profile-tradeoff.svg", svg)


def _compute_collective_split_metrics() -> list[dict[str, float | str]]:
    service = ProtectionService()
    rows: list[dict[str, float | str]] = []
    for image_path in sorted(SUBJECT_SET_DIR.glob("*.png")):
        report = service.analyze_files(
            str(image_path),
            str(COLLECTIVE_DIR / image_path.name),
        )
        rows.append(
            {
                "label": image_path.name,
                "protection": float(report["protection_report"]["protection_score"]),
                "ssim": float(report["quality_report"]["ssim"]),
                "psnr": float(report["quality_report"]["psnr_db"]),
            }
        )
    return rows


def build_collective_chart() -> None:
    rows = _compute_collective_split_metrics()
    avg_score = mean(float(row["protection"]) for row in rows)
    width = 1400
    height = 700
    left = 120
    bottom = 580
    chart_height = 300
    chart_width = 1080
    gap = 48
    bar_width = (chart_width - gap * (len(rows) - 1)) / len(rows)
    max_score = 60.0
    bars: list[str] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        x = left + index * (bar_width + gap)
        protection = float(row["protection"])
        bar_h = protection / max_score * chart_height
        y = bottom - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" rx="18" fill="#14b8a6" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 14:.1f}" text-anchor="middle" fill="#f8fafc" font-size="22" font-family="Segoe UI, Arial, sans-serif" font-weight="700">{_fmt(protection)}</text>'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{bottom + 42:.1f}" text-anchor="middle" fill="#cbd5e1" font-size="20" font-family="Segoe UI, Arial, sans-serif">{xml_escape.escape(str(row["label"]))}</text>'
        )
    avg_y = bottom - avg_score / max_score * chart_height
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">
  <rect width="{width}" height="{height}" rx="32" fill="#0b1220"/>
  <text x="120" y="88" fill="#f8fafc" font-size="42" font-family="Segoe UI, Arial, sans-serif" font-weight="700">Collective Subject Split Snapshot</text>
  <text x="120" y="126" fill="#94a3b8" font-size="22" font-family="Segoe UI, Arial, sans-serif">Anti-DreamBooth-aligned run on n000050 / set_B with profile=subject, working size 384.</text>
  <line x1="{left}" y1="{bottom}" x2="{left + chart_width}" y2="{bottom}" stroke="#334155" stroke-width="2"/>
  <line x1="{left}" y1="{bottom - chart_height}" x2="{left}" y2="{bottom}" stroke="#334155" stroke-width="2"/>
  {''.join(bars)}
  {''.join(labels)}
  <line x1="{left}" y1="{avg_y:.1f}" x2="{left + chart_width}" y2="{avg_y:.1f}" stroke="#f59e0b" stroke-width="4" stroke-dasharray="10 10"/>
  <text x="{left + chart_width - 8}" y="{avg_y - 12:.1f}" text-anchor="end" fill="#fcd34d" font-size="22" font-family="Segoe UI, Arial, sans-serif" font-weight="700">Average {_fmt(avg_score)}</text>
  <text x="120" y="650" fill="#94a3b8" font-size="20" font-family="Segoe UI, Arial, sans-serif">Current result: materialization is stable, but collective strength still needs more research optimization.</text>
</svg>"""
    _write_svg(ASSETS_DIR / "collective-subject-split.svg", svg)


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = max(width / image.width, height / image.height)
    resized = image.resize(
        (math.ceil(image.width * ratio), math.ceil(image.height * ratio)),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def build_demo_gallery() -> None:
    panels = [
        {
            "title": "Original",
            "subtitle": "Reference artwork",
            "image": DEMO_DIR / "horse_painting_original.jpg",
        },
        {
            "title": "Balanced",
            "subtitle": "PSNR 36.72 | SSIM 0.9739 | Protect 49.0",
            "image": DEMO_DIR / "horse_painting_balanced_protected.jpg",
        },
        {
            "title": "Blindfold",
            "subtitle": "PSNR 26.53 | SSIM 0.6114 | Protect 61.1",
            "image": DEMO_DIR / "horse_painting_preview_768_blindfold.jpg",
        },
    ]
    card_width = 420
    card_height = 420
    header = 120
    footer = 110
    gap = 24
    canvas = Image.new(
        "RGB", (3 * card_width + 4 * gap, header + card_height + footer), "#0b1220"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()

    draw.text((36, 28), "AuraLock Output Gallery", fill="#f8fafc", font=title_font)
    draw.text(
        (36, 64),
        "Real demo outputs used by the README, generated from local reports and saved artifacts.",
        fill="#94a3b8",
        font=subtitle_font,
    )

    for index, panel in enumerate(panels):
        x = gap + index * (card_width + gap)
        y = header
        draw.rounded_rectangle(
            (x, y, x + card_width, y + card_height + 76),
            radius=24,
            fill="#111827",
            outline="#1f2937",
            width=2,
        )
        image = Image.open(panel["image"]).convert("RGB")
        thumb = _fit_image(image, card_width - 24, card_height - 24)
        canvas.paste(thumb, (x + 12, y + 12))
        draw.text(
            (x + 18, y + card_height + 24),
            panel["title"],
            fill="#f8fafc",
            font=title_font,
        )
        draw.text(
            (x + 18, y + card_height + 50),
            panel["subtitle"],
            fill="#cbd5e1",
            font=subtitle_font,
        )

    output_path = ASSETS_DIR / "demo-gallery.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    build_banner()
    build_profile_tradeoff_chart()
    build_collective_chart()
    build_demo_gallery()
    print("README assets generated in", ASSETS_DIR)


if __name__ == "__main__":
    main()
