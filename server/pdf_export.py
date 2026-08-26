"""
PDF export for summaries (meeting minutes etc.).

Renders the stored Markdown summary into a clean A4 PDF with reportlab.
Font strategy: DejaVu Sans when available (good Latin/Cyrillic coverage),
and reportlab's built-in Japanese CID font for CJK-heavy content - no font
files need to be shipped either way.
"""
import html
import io
import os
import re
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable, ListFlowable, ListItem)

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# CJK-capable fonts to EMBED, searched in order. Embedding matters: the
# fallback Adobe CID font is not embedded, so viewers without Japanese font
# packs render blanks.
CJK_FONT_CANDIDATES = [
    # Windows
    (r"C:\Windows\Fonts\meiryo.ttc", 0),
    (r"C:\Windows\Fonts\YuGothM.ttc", 0),
    (r"C:\Windows\Fonts\msgothic.ttc", 0),
    # Linux
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc", 0),
    # macOS
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
]

CJK_RE = re.compile(r"[　-ヿ㐀-鿿＀-￯]")

_registered = {}


def _register_ttf(name, path, subfont=None):
    if name in _registered:
        return _registered[name]
    kwargs = {"subfontIndex": subfont} if subfont is not None else {}
    pdfmetrics.registerFont(TTFont(name, path, **kwargs))
    _registered[name] = name
    return name


def _pick_fonts(text):
    """Return (body_font, bold_font) able to render `text`, preferring
    embedded TTFs so the PDF looks the same on every machine."""
    if CJK_RE.search(text or ""):
        for path, subfont in CJK_FONT_CANDIDATES:
            try:
                if os.path.exists(path):
                    font = _register_ttf("EchoCJK", path, subfont)
                    return font, font
            except Exception:
                continue
        # Last resort: Adobe CID font (not embedded - needs a viewer with
        # Japanese font support, e.g. Acrobat's Asian font pack)
        if "HeiseiKakuGo-W5" not in _registered:
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
            _registered["HeiseiKakuGo-W5"] = "HeiseiKakuGo-W5"
        return "HeiseiKakuGo-W5", "HeiseiKakuGo-W5"
    try:
        _register_ttf("DejaVu", DEJAVU)
        _register_ttf("DejaVu-Bold", DEJAVU_BOLD)
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _inline(text):
    """Escape + minimal inline markdown (**bold**) for reportlab paragraphs."""
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def build_summary_pdf(recording, summary_md, template_name="summary"):
    """Render a summary's Markdown to PDF bytes."""
    body_font, bold_font = _pick_fonts((recording.get("title") or "") + summary_md)

    styles = {
        "title": ParagraphStyle("title", fontName=bold_font, fontSize=18, leading=23,
                                spaceAfter=2, textColor=colors.HexColor("#1a1a1e")),
        "meta": ParagraphStyle("meta", fontName=body_font, fontSize=9, leading=12,
                               textColor=colors.HexColor("#71717a"), spaceAfter=10),
        "h": ParagraphStyle("h", fontName=bold_font, fontSize=13, leading=17,
                            spaceBefore=12, spaceAfter=4,
                            textColor=colors.HexColor("#166534")),
        "body": ParagraphStyle("body", fontName=body_font, fontSize=10.5, leading=15,
                               spaceAfter=4),
        "li": ParagraphStyle("li", fontName=body_font, fontSize=10.5, leading=15),
        "footer": ParagraphStyle("footer", fontName=body_font, fontSize=8, leading=10,
                                 textColor=colors.HexColor("#a1a1aa")),
    }

    story = [Paragraph(_inline(recording.get("title") or "Recording"), styles["title"])]

    created = recording.get("created_at")
    meta_bits = []
    if created:
        meta_bits.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(created)))
    if recording.get("duration_s"):
        m, s = divmod(int(recording["duration_s"]), 60)
        meta_bits.append(f"{m} min {s} s" if m else f"{s} s")
    if recording.get("language"):
        meta_bits.append(recording["language"])
    meta_bits.append(template_name.replace("-", " "))
    story.append(Paragraph(" · ".join(meta_bits), styles["meta"]))
    story.append(HRFlowable(width="100%", thickness=0.7,
                            color=colors.HexColor("#d4d4d8"), spaceAfter=8))

    bullets = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(b, styles["li"]), leftIndent=14) for b in bullets],
                bulletType="bullet", bulletFontName=body_font, bulletFontSize=9,
                leftIndent=14, spaceBefore=2, spaceAfter=4))
            bullets = []

    for raw in (summary_md or "").splitlines():
        line = raw.rstrip()
        m_h = re.match(r"^#{1,3}\s+(.*)", line)
        m_check = re.match(r"^\s*[-*]\s*\[([ xX]?)\]\s*(.*)", line)
        m_bullet = re.match(r"^\s*[-*]\s+(.*)", line)
        if m_h:
            flush_bullets()
            story.append(Paragraph(_inline(m_h.group(1)), styles["h"]))
        elif m_check:
            mark = "[x]" if m_check.group(1).strip() else "[ ]"
            bullets.append(f"{mark}  {_inline(m_check.group(2))}")
        elif m_bullet:
            bullets.append(_inline(m_bullet.group(1)))
        elif not line.strip():
            flush_bullets()
        else:
            flush_bullets()
            # bare "**Section**" lines act as headers in some model output
            m_bold = re.match(r"^\s*\*\*([^*]+)\*\*:?\s*$", line)
            if m_bold:
                story.append(Paragraph(_inline(m_bold.group(1)), styles["h"]))
            else:
                story.append(Paragraph(_inline(line), styles["body"]))
    flush_bullets()

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#e4e4e7"), spaceAfter=4))
    story.append(Paragraph("Generated locally by EchoNote", styles["footer"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=recording.get("title") or "EchoNote summary",
        author="EchoNote")
    doc.build(story)
    return buf.getvalue()
