from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_MD = ROOT / "SLIDE_BAO_VE.md"
OUTPUT_HTML = ROOT / "SLIDE_BAO_VE.html"


MEDIA_MAP = {
    "1": {
        "items": [
            {
                "src": "images/VGG16/VGG16-11.png",
                "caption": "Ví dụ trực quan: ảnh đầu vào và caption sinh ra",
            }
        ]
    },
    "5": {
        "items": [
            {
                "src": "images/VGG16/VGG16-1.png",
                "caption": "Phân tích kích thước ảnh trong Flickr30k",
            },
            {
                "src": "images/VGG16/VGG16-6.png",
                "caption": "Thống kê split train / validation / test",
            },
            {
                "src": "images/CLIP/clip1.png",
                "caption": "EDA tổng quan từ notebook CLIP",
            },
        ]
    },
    "6": {
        "items": [
            {
                "src": "images/VGG16/VGG16-4.png",
                "caption": "Caption cleaning trước và sau xử lý",
            },
            {
                "src": "images/VGG16/VGG16-5.png",
                "caption": "Vocabulary cutoff và GloVe coverage",
            },
            {
                "src": "images/Resnet/Resnet3.png",
                "caption": "Thống kê caption cleaning từ notebook ResNet",
            },
            {
                "src": "images/CLIP/clip4.png",
                "caption": "Tokenization và GloVe embeddings cho CLIP",
            },
        ]
    },
    "7": {
        "items": [
            {
                "src": "images/VGG16/VGG16-7.png",
                "caption": "Kiến trúc mô hình và attention map 7×7",
            }
        ]
    },
    "8": {
        "items": [
            {
                "src": "images/VGG16/VGG16-3.png",
                "caption": "VGG16 feature verification (shape: 49×512)",
            }
        ]
    },
    "9": {
        "items": [
            {
                "src": "images/Resnet/Resnet2.png",
                "caption": "ResNet-101 feature verification (shape: 49×2048)",
            }
        ]
    },
    "10": {
        "items": [
            {
                "src": "images/CLIP/clip2.png",
                "caption": "CLIP ViT-B/32 patch features (shape: 49×512)",
            }
        ]
    },
    "11": {
        "items": [
            {
                "src": "images/VGG16/VGG16-7.png",
                "caption": "Attention map trực quan trong decoder",
            }
        ]
    },
    "13": {
        "items": [
            {
                "src": "images/VGG16/VGG16-8.png",
                "caption": "Training results của VGG16",
            },
            {
                "src": "images/Resnet/Resnet4.png",
                "caption": "Training history của ResNet-101",
            },
            {
                "src": "images/CLIP/clip5.png",
                "caption": "Training results của CLIP ViT-B/32",
            },
        ]
    },
    "14": {
        "items": [
            {
                "src": "images/VGG16/VGG16-9.png",
                "caption": "BLEU evaluation của VGG16",
            },
            {
                "src": "images/Resnet/Resnet5.png",
                "caption": "BLEU evaluation của ResNet-101",
            },
            {
                "src": "images/CLIP/clip6.png",
                "caption": "BLEU evaluation của CLIP",
            },
        ]
    },
    "15": {
        "items": [
            {
                "src": "images/VGG16/VGG16-10.png",
                "caption": "Generated caption analysis của VGG16",
            },
            {
                "src": "images/Resnet/Resnet6.png",
                "caption": "Generated caption analysis của ResNet-101",
            },
        ]
    },
    "16": {
        "items": [
            {
                "src": "images/VGG16/VGG16-12.png",
                "caption": "Demo caption generation của VGG16",
            },
            {
                "src": "images/Resnet/Resnet7.png",
                "caption": "Demo caption generation của ResNet-101",
            },
            {
                "src": "images/CLIP/clip7.png",
                "caption": "Demo caption generation của CLIP",
            },
        ]
    },
    "17": {
        "items": [
            {
                "src": "images/Resnet/Resnet6.png",
                "caption": "Phân tích chất lượng caption sinh ra",
            }
        ]
    },
}


def load_source() -> str:
    return SOURCE_MD.read_text(encoding="utf-8")


def extract_slides(markdown: str):
    lines = markdown.splitlines()
    slides = []
    preamble = []
    current = None
    collecting_note = False

    heading_re = re.compile(r"^(# SLIDE \d+\s+—\s+.+|# PHỤ LỤC.+|## ❓\s+Q\d+:.+)$")

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if heading_re.match(line.strip()):
            if current:
                slides.append(current)
            current = {"title_line": line.strip(), "content": [], "notes": []}
            collecting_note = False
            continue

        if current is None:
            if line.strip() and line.strip() != "---":
                preamble.append(line)
            continue

        if line.strip() == "---":
            collecting_note = False
            continue

        if line.startswith("> 🗣️"):
            collecting_note = True
            current["notes"].append(line[2:].strip())
            continue

        if collecting_note and line.startswith("> "):
            current["notes"][-1] += "\n" + line[2:].rstrip()
            continue

        collecting_note = False
        current["content"].append(line)

    if current:
        slides.append(current)

    return preamble, slides


def inline_format(text: str) -> str:
    text = text.replace("&nbsp;", "\u00a0")
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def parse_table(block_lines: list[str]) -> str:
    rows = []
    for line in block_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return ""

    header = rows[0]
    body = rows[2:] if re.fullmatch(r"[:\- ]+", rows[1][0].replace("|", "")) else rows[1:]

    parts = ['<table class="md-table">', "<thead><tr>"]
    for cell in header:
        parts.append(f"<th>{inline_format(cell)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{inline_format(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_blockquote(block_lines: list[str]) -> str:
    cleaned = []
    for line in block_lines:
        cleaned.append(inline_format(line.lstrip("> ").strip()))
    return '<blockquote class="md-quote">' + "<br>".join(cleaned) + "</blockquote>"


def render_list(block_lines: list[str], ordered: bool) -> str:
    tag = "ol" if ordered else "ul"
    items = []
    for line in block_lines:
        if ordered:
            content = re.sub(r"^\d+\.\s*", "", line).strip()
        else:
            content = re.sub(r"^-\s*", "", line).strip()
        items.append(f"<li>{inline_format(content)}</li>")
    return f'<{tag} class="md-list">{"".join(items)}</{tag}>'


def render_markdown(lines: list[str]) -> str:
    html_parts = []
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "<br>":
            html_parts.append('<div class="spacer"></div>')
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < total and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            html_parts.append(
                '<pre class="md-code"><code>'
                + html.escape("\n".join(code_lines))
                + "</code></pre>"
            )
            i += 1
            continue

        if stripped.startswith("|"):
            block = []
            while i < total and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            html_parts.append(parse_table(block))
            continue

        if stripped.startswith(">"):
            block = []
            while i < total and lines[i].strip().startswith(">"):
                block.append(lines[i].strip())
                i += 1
            html_parts.append(render_blockquote(block))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            block = []
            while i < total and re.match(r"^\d+\.\s+", lines[i].strip()):
                block.append(lines[i].strip())
                i += 1
            html_parts.append(render_list(block, ordered=True))
            continue

        if stripped.startswith("- "):
            block = []
            while i < total and lines[i].strip().startswith("- "):
                block.append(lines[i].strip())
                i += 1
            html_parts.append(render_list(block, ordered=False))
            continue

        if stripped.startswith("### "):
            html_parts.append(f'<h3 class="md-h3">{inline_format(stripped[4:])}</h3>')
            i += 1
            continue

        if stripped.startswith("## "):
            html_parts.append(f'<h2 class="md-h2">{inline_format(stripped[3:])}</h2>')
            i += 1
            continue

        if stripped.startswith("# "):
            html_parts.append(f'<h1 class="md-h1">{inline_format(stripped[2:])}</h1>')
            i += 1
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < total:
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt == "<br>"
                or nxt.startswith("|")
                or nxt.startswith("```")
                or nxt.startswith(">")
                or nxt.startswith("#")
                or nxt.startswith("- ")
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            paragraph_lines.append(nxt)
            i += 1

        paragraph = " ".join(paragraph_lines)
        klass = "md-callout" if paragraph[:1] in {"✅", "📌", "⚠", "📖", "❓"} else "md-p"
        html_parts.append(f'<p class="{klass}">{inline_format(paragraph)}</p>')

    return "\n".join(html_parts)


def slide_meta(title_line: str) -> dict[str, str]:
    slide_match = re.match(r"^# SLIDE (\d+)\s+—\s+(.+)$", title_line)
    if slide_match:
        number, label = slide_match.groups()
        return {"key": number, "chip": f"SLIDE {number}", "title": label}

    appendix_match = re.match(r"^## ❓\s+(Q\d+):\s+(.+)$", title_line)
    if appendix_match:
        key, label = appendix_match.groups()
        return {"key": key, "chip": "PHỤ LỤC", "title": f"{key}: {label}"}

    return {"key": "PHU_LUC", "chip": "PHỤ LỤC", "title": title_line.replace("# ", "", 1)}


def render_media(key: str) -> tuple[str, str]:
    media = MEDIA_MAP.get(key, {})
    items = media.get("items", [])
    if not items:
        return "", "text-only"

    layout = "side" if len(items) == 1 else "bottom"
    figure_parts = []
    for item in items:
        figure_parts.append(
            f"""
            <figure class="media-card">
              <img src="{html.escape(item['src'])}" alt="{html.escape(item['caption'])}">
              <figcaption>{html.escape(item['caption'])}</figcaption>
            </figure>
            """
        )

    return f'<div class="media-grid media-grid--{layout}">{"".join(figure_parts)}</div>', layout


def build_html(preamble: list[str], slides: list[dict]) -> str:
    slide_sections = []

    for index, slide in enumerate(slides, start=1):
        meta = slide_meta(slide["title_line"])
        body_html = render_markdown(slide["content"])
        media_html, layout = render_media(meta["key"])
        notes_html = "<br>".join(html.escape(note) for note in slide["notes"]) if slide["notes"] else "Không có lời thoại riêng cho slide này."

        slide_classes = ["slide-card", f"layout-{layout}"]
        if meta["key"] in {"1", "20"}:
            slide_classes.append("slide-card--hero")
        if meta["key"] == "2":
            slide_classes.append("slide-card--agenda")
        if meta["key"] == "PHU_LUC":
            slide_classes.append("slide-card--appendix")

        section = f"""
        <section class="slide {' '.join(slide_classes)}" data-slide-index="{index}" data-slide-key="{meta['key']}" data-notes="{html.escape(notes_html)}">
          <div class="slide-topbar">
            <div class="slide-chip">{html.escape(meta['chip'])}</div>
            <div class="slide-heading">{html.escape(meta['title'])}</div>
          </div>
          <div class="slide-content">
            <div class="slide-text">{body_html}</div>
            {media_html}
          </div>
          <div class="slide-footer">
            <span>Neural Image Caption Generation</span>
            <span>{index} / {len(slides)}</span>
          </div>
        </section>
        """
        slide_sections.append(section)

    preamble_html = "<br>".join(html.escape(line.strip()) for line in preamble if line.strip())

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SLIDE_BAO_VE - Neural Image Caption Generation</title>
  <style>
    :root {{
      --bg: #0b1c33;
      --panel: #ffffff;
      --panel-soft: #f4f7fb;
      --ink: #10243c;
      --muted: #5f7289;
      --brand: #143d72;
      --brand-2: #2f73be;
      --accent: #f08a2b;
      --accent-soft: #fff2e4;
      --line: rgba(16, 36, 60, 0.12);
      --shadow: 0 34px 90px rgba(5, 18, 37, 0.28);
      --radius: 28px;
      --viewport-pad: clamp(12px, 2vw, 28px);
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(62, 134, 214, 0.22), transparent 32%),
        radial-gradient(circle at top right, rgba(240, 138, 43, 0.18), transparent 28%),
        linear-gradient(145deg, #091427 0%, #0b1c33 48%, #122949 100%);
      color: var(--ink);
    }}

    body {{
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}

    .app-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px var(--viewport-pad) 10px;
      color: #eef5ff;
    }}

    .app-title {{
      display: grid;
      gap: 4px;
    }}

    .app-title strong {{
      font-size: clamp(1rem, 1.8vw, 1.2rem);
      letter-spacing: 0.03em;
    }}

    .app-title span {{
      font-size: 0.9rem;
      color: rgba(238, 245, 255, 0.78);
    }}

    .header-tools {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .chip {{
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.85rem;
      background: rgba(255, 255, 255, 0.08);
      color: #f7fbff;
      backdrop-filter: blur(16px);
    }}

    .viewport {{
      position: relative;
      display: grid;
      place-items: center;
      padding: var(--viewport-pad);
    }}

    .deck {{
      position: relative;
      width: min(96vw, calc(93vh * 16 / 9));
      aspect-ratio: 16 / 9;
    }}

    .slide {{
      position: absolute;
      inset: 0;
      display: none;
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: var(--shadow);
      background: var(--panel);
    }}

    .slide.is-active {{
      display: grid;
      grid-template-rows: 88px 1fr 44px;
      animation: fadeIn 220ms ease;
    }}

    .slide-card--hero {{
      background:
        radial-gradient(circle at top right, rgba(240, 138, 43, 0.18), transparent 30%),
        linear-gradient(145deg, #0f2950 0%, #143d72 42%, #0e2141 100%);
      color: #f6fbff;
    }}

    .slide-card--hero .slide-topbar,
    .slide-card--hero .slide-footer {{
      background: transparent;
      border-color: rgba(255, 255, 255, 0.12);
      color: #eef5ff;
    }}

    .slide-card--hero .slide-chip {{
      background: rgba(255, 255, 255, 0.12);
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.18);
    }}

    .slide-card--hero .slide-heading,
    .slide-card--hero .slide-text,
    .slide-card--hero .md-h1,
    .slide-card--hero .md-h2,
    .slide-card--hero .md-h3,
    .slide-card--hero .md-p,
    .slide-card--hero .md-callout,
    .slide-card--hero .md-list,
    .slide-card--hero .md-table,
    .slide-card--hero .md-quote {{
      color: #f6fbff;
    }}

    .slide-card--hero .md-table td,
    .slide-card--hero .md-table th {{
      border-color: rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.05);
    }}

    .slide-card--hero code,
    .slide-card--hero .md-code {{
      background: rgba(3, 11, 24, 0.35);
      color: #ffe3c1;
    }}

    .slide-topbar,
    .slide-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 26px;
      background: linear-gradient(180deg, rgba(20, 61, 114, 0.98), rgba(16, 47, 87, 0.98));
      color: #eff5ff;
    }}

    .slide-footer {{
      background: #13365f;
      font-size: 0.84rem;
      color: rgba(239, 245, 255, 0.88);
    }}

    .slide-chip {{
      flex: 0 0 auto;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}

    .slide-heading {{
      font-size: clamp(1rem, 1.75vw, 1.28rem);
      font-weight: 700;
      text-align: right;
      letter-spacing: 0.01em;
    }}

    .slide-content {{
      min-height: 0;
      padding: 22px 26px 20px;
      display: grid;
      gap: 18px;
    }}

    .layout-text-only .slide-content {{
      grid-template-columns: 1fr;
    }}

    .layout-side .slide-content {{
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      align-items: stretch;
    }}

    .layout-bottom .slide-content {{
      grid-template-rows: minmax(0, 1fr) minmax(220px, 0.9fr);
    }}

    .slide-text {{
      min-height: 0;
      overflow: auto;
      padding-right: 8px;
      scrollbar-width: thin;
      scrollbar-color: rgba(20, 61, 114, 0.4) transparent;
    }}

    .slide-text::-webkit-scrollbar {{
      width: 8px;
    }}

    .slide-text::-webkit-scrollbar-thumb {{
      background: rgba(20, 61, 114, 0.35);
      border-radius: 999px;
    }}

    .md-h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 3.8vw, 3rem);
      line-height: 1.05;
      letter-spacing: -0.04em;
      color: var(--brand);
    }}

    .md-h2 {{
      margin: 0 0 12px;
      font-size: clamp(1.18rem, 2vw, 1.55rem);
      line-height: 1.18;
      color: var(--brand);
    }}

    .md-h3 {{
      margin: 18px 0 10px;
      font-size: clamp(1rem, 1.6vw, 1.15rem);
      color: var(--brand-2);
    }}

    .md-p,
    .md-callout,
    .md-list,
    .md-quote {{
      margin: 0 0 12px;
      font-size: clamp(0.88rem, 1.23vw, 1.02rem);
      line-height: 1.48;
      color: var(--ink);
    }}

    .md-callout {{
      padding: 10px 14px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      border-radius: 12px;
      font-weight: 600;
    }}

    .md-list {{
      padding-left: 24px;
    }}

    .md-list li {{
      margin-bottom: 8px;
    }}

    .md-quote {{
      padding: 12px 16px;
      background: #eef5ff;
      border-left: 4px solid var(--brand-2);
      border-radius: 14px;
      color: #244362;
    }}

    .md-code {{
      margin: 0 0 12px;
      padding: 14px 16px;
      border-radius: 16px;
      background: #0e1b2c;
      color: #f7fafc;
      overflow: auto;
      font-size: clamp(0.78rem, 1.04vw, 0.92rem);
      line-height: 1.4;
    }}

    code {{
      padding: 0.14em 0.36em;
      border-radius: 8px;
      background: rgba(20, 61, 114, 0.08);
      color: #12335c;
      font-size: 0.92em;
    }}

    .md-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0 0 12px;
      border-radius: 16px;
      overflow: hidden;
      font-size: clamp(0.78rem, 1.02vw, 0.92rem);
    }}

    .md-table th,
    .md-table td {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      vertical-align: top;
    }}

    .md-table thead th {{
      background: #173d70;
      color: #f3f7ff;
      text-align: left;
    }}

    .md-table tbody tr:nth-child(odd) td {{
      background: #f6f9fc;
    }}

    .spacer {{
      height: 14px;
    }}

    .media-grid {{
      min-height: 0;
      display: grid;
      gap: 14px;
    }}

    .media-grid--side {{
      grid-template-columns: 1fr;
      align-content: stretch;
    }}

    .media-grid--bottom {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      align-content: start;
      overflow: auto;
      padding-right: 6px;
    }}

    .media-card {{
      margin: 0;
      background: #f8fbff;
      border: 1px solid rgba(18, 51, 92, 0.1);
      border-radius: 18px;
      overflow: hidden;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      box-shadow: 0 14px 28px rgba(10, 33, 60, 0.08);
    }}

    .media-card img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #ffffff;
      display: block;
    }}

    .media-card figcaption {{
      padding: 10px 12px 12px;
      font-size: 0.82rem;
      line-height: 1.35;
      color: var(--muted);
      background: #f6faff;
      border-top: 1px solid rgba(18, 51, 92, 0.08);
    }}

    .notes-panel {{
      position: fixed;
      top: 84px;
      right: 18px;
      width: min(360px, calc(100vw - 36px));
      max-height: calc(100vh - 150px);
      overflow: auto;
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(7, 18, 34, 0.88);
      color: #f7fbff;
      box-shadow: 0 24px 48px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(20px);
      display: none;
      z-index: 20;
    }}

    .notes-panel.is-open {{
      display: block;
    }}

    .notes-panel h3 {{
      margin: 0 0 10px;
      font-size: 0.96rem;
      letter-spacing: 0.04em;
      color: #ffd3a2;
    }}

    .notes-panel p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 0.92rem;
    }}

    .progress-bar {{
      height: 6px;
      margin: 0 var(--viewport-pad) 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      overflow: hidden;
    }}

    .progress-bar span {{
      display: block;
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #55a4ff 0%, #f08a2b 100%);
      transition: width 180ms ease;
    }}

    .help-panel {{
      position: fixed;
      left: 18px;
      bottom: 18px;
      max-width: min(420px, calc(100vw - 36px));
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.08);
      color: #eef5ff;
      font-size: 0.88rem;
      line-height: 1.45;
      backdrop-filter: blur(18px);
    }}

    .help-panel strong {{
      display: block;
      margin-bottom: 4px;
      color: #ffffff;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    @media (max-width: 980px) {{
      .deck {{
        width: 100%;
        height: 100%;
        aspect-ratio: auto;
      }}

      .slide.is-active {{
        grid-template-rows: 74px 1fr 42px;
      }}

      .layout-side .slide-content,
      .layout-bottom .slide-content {{
        grid-template-columns: 1fr;
        grid-template-rows: auto;
      }}

      .slide-topbar,
      .slide-footer {{
        padding: 0 18px;
      }}

      .slide-content {{
        padding: 16px 18px;
      }}
    }}
  </style>
</head>
<body>
  <header class="app-header">
    <div class="app-title">
      <strong>SLIDE BẢO VỆ ĐỒ ÁN</strong>
      <span>Neural Image Caption Generation</span>
    </div>
    <div class="header-tools">
      <div class="chip">← → chuyển slide</div>
      <div class="chip">N mở lời thoại</div>
      <div class="chip">F fullscreen</div>
    </div>
  </header>

  <main class="viewport">
    <div class="deck" id="deck">
      {''.join(slide_sections)}
    </div>
  </main>

  <div class="progress-bar" aria-hidden="true"><span id="progress"></span></div>

  <aside class="notes-panel" id="notesPanel">
    <h3>LỜI THOẠI TRÌNH BÀY</h3>
    <p id="notesBody"></p>
  </aside>

  <div class="help-panel">
    <strong>Ghi chú</strong>
    <div>{preamble_html}</div>
  </div>

  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    const progress = document.getElementById('progress');
    const notesPanel = document.getElementById('notesPanel');
    const notesBody = document.getElementById('notesBody');
    let current = 0;

    function updateSlide(nextIndex) {{
      current = Math.max(0, Math.min(nextIndex, slides.length - 1));
      slides.forEach((slide, index) => {{
        slide.classList.toggle('is-active', index === current);
      }});

      const active = slides[current];
      notesBody.innerHTML = active.dataset.notes || 'Không có lời thoại riêng cho slide này.';
      progress.style.width = `${{((current + 1) / slides.length) * 100}}%`;
      document.title = `${{active.querySelector('.slide-chip').textContent}} - ${{active.querySelector('.slide-heading').textContent}}`;
    }}

    function nextSlide() {{
      updateSlide(current + 1);
    }}

    function previousSlide() {{
      updateSlide(current - 1);
    }}

    document.addEventListener('keydown', (event) => {{
      if (['ArrowRight', 'PageDown', ' '].includes(event.key)) {{
        event.preventDefault();
        nextSlide();
      }} else if (['ArrowLeft', 'PageUp'].includes(event.key)) {{
        event.preventDefault();
        previousSlide();
      }} else if (event.key === 'Home') {{
        event.preventDefault();
        updateSlide(0);
      }} else if (event.key === 'End') {{
        event.preventDefault();
        updateSlide(slides.length - 1);
      }} else if (event.key.toLowerCase() === 'n') {{
        event.preventDefault();
        notesPanel.classList.toggle('is-open');
      }} else if (event.key.toLowerCase() === 'f') {{
        event.preventDefault();
        if (!document.fullscreenElement) {{
          document.documentElement.requestFullscreen?.();
        }} else {{
          document.exitFullscreen?.();
        }}
      }}
    }});

    document.addEventListener('click', (event) => {{
      if (event.target.closest('.notes-panel')) return;
      if (event.clientX > window.innerWidth * 0.68) {{
        nextSlide();
      }} else if (event.clientX < window.innerWidth * 0.32) {{
        previousSlide();
      }}
    }});

    updateSlide(0);
  </script>
</body>
</html>
"""


def main() -> None:
    markdown = load_source()
    preamble, slides = extract_slides(markdown)
    html_output = build_html(preamble, slides)
    OUTPUT_HTML.write_text(html_output, encoding="utf-8")
    print(f"Da tao: {OUTPUT_HTML.name}")
    print(f"So slide: {len(slides)}")


if __name__ == "__main__":
    main()
