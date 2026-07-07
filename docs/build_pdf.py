"""
Combine the three documentation markdown files into a single PDF, with a cover
page and the rendered system-architecture diagram.

Pipeline: markdown -> HTML (python-markdown) -> PDF (xhtml2pdf/pisa).
Fonts: DejaVu Sans / Sans Mono (shipped with matplotlib) are embedded so that
box-drawing characters and arrows in the ASCII diagrams render correctly.

Run:  python docs/build_pdf.py
Out:  docs/Railway_PdM_Documentation.pdf
"""
from __future__ import annotations

import datetime as _dt
import os

import markdown as _md
from xhtml2pdf import pisa

HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAM_PNG = os.path.join(HERE, "architecture_diagram.png")
OUT_PDF = os.path.join(HERE, "Railway_PdM_Documentation.pdf")


def _uri(path: str) -> str:
    """file:// URI with forward slashes. The explicit scheme stops xhtml2pdf
    from mistaking a Windows drive letter (e.g. 'D:') for a URL scheme."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def _link_callback(uri: str, rel: str) -> str:
    """Resolve font/image URIs to real local file paths for xhtml2pdf."""
    path = uri
    if path.startswith("file:///"):
        path = path[len("file:///"):]
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"asset not found for URI '{uri}' -> '{path}'")
    return path

# Order the sections appear in the combined document.
DOCS = [
    ("Beginner's Guide", "BEGINNERS_GUIDE.md"),
    ("Model Evaluation & Benchmarks", "EVALUATION.md"),
    ("MRO / Asset-Management Integration Roadmap", "INTEGRATION_ROADMAP.md"),
]

# matplotlib-bundled DejaVu fonts (broad glyph coverage incl. box-drawing).
# The built-in Helvetica/Courier fonts cover WinAnsi (incl. em-dash, deg, x,
# section, superscript-2). A handful of glyphs used in the docs fall outside
# WinAnsi (arrows, box-drawing, >=, checkmark); we map those to ASCII so the
# PDF renders cleanly without embedding external fonts (xhtml2pdf's @font-face
# loader is unreliable on Windows). The polished architecture visual is carried
# by the rendered PNG diagram, not the ASCII art.
_GLYPH_MAP = {
    "→": "->", "←": "<-", "↔": "<->",   # arrows
    "▼": "v", "▶": ">",                       # triangles
    "≥": ">=", "≤": "<=",                     # comparisons
    "✅": "[OK]", "✓": "[OK]",                 # check marks
    "─": "-", "│": "|",                       # box: horiz / vert
    "┌": "+", "┐": "+", "└": "+", "┘": "+",  # corners
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
}


def _sanitize(text: str) -> str:
    for src, dst in _GLYPH_MAP.items():
        text = text.replace(src, dst)
    return text


CSS = f"""
@page {{ size: letter; margin: 1.7cm 1.6cm 1.9cm 1.6cm; }}

body {{ font-family: Helvetica; font-size: 10pt; line-height: 1.4; color: #1a1a1a; }}
h1 {{ font-size: 19pt; color: #C5283D; margin: 4pt 0 8pt 0; }}
h2 {{ font-size: 14.5pt; color: #6A4C93; margin: 14pt 0 5pt 0;
      border-bottom: 1px solid #ddd; padding-bottom: 2pt; }}
h3 {{ font-size: 11.5pt; color: #2E86AB; margin: 10pt 0 3pt 0; }}
h4 {{ font-size: 10.5pt; color: #444; margin: 8pt 0 2pt 0; }}
p  {{ margin: 4pt 0; }}
a  {{ color: #2E86AB; text-decoration: none; }}
ul, ol {{ margin: 3pt 0 3pt 0; }}
li {{ margin: 1.5pt 0; }}

code {{ font-family: Courier; font-size: 8.6pt; background: #f2f2f4; padding: 0 2px; }}
pre {{ font-family: Courier; font-size: 7.8pt; background: #f6f6f8;
       border: 1px solid #e2e2e6; padding: 6pt; margin: 5pt 0; line-height: 1.25; }}
pre code {{ background: transparent; padding: 0; }}

table {{ border-collapse: collapse; margin: 6pt 0; width: 100%; }}
th {{ background: #6A4C93; color: #ffffff; font-size: 8.4pt; padding: 4pt 5pt;
      text-align: left; border: 0.5pt solid #6A4C93; }}
td {{ font-size: 8.4pt; padding: 3.5pt 5pt; border: 0.5pt solid #cfcfd6;
      vertical-align: top; }}
tr:nth-child(even) td {{ background: #f4f2f8; }}

.cover {{ text-align: center; }}
.cover h1 {{ font-size: 27pt; color: #C5283D; border: 0; margin-top: 130pt; }}
.cover .sub {{ font-size: 13pt; color: #555; margin-top: 8pt; }}
.cover .toc {{ font-size: 11pt; color: #333; margin-top: 46pt; line-height: 1.7; }}
.cover .toc b {{ font-size: 12pt; color: #6A4C93; }}
.cover .meta {{ font-size: 10pt; color: #888; margin-top: 40pt; }}

.arch {{ text-align: center; }}
.arch img {{ width: 100%; }}
.arch .cap {{ font-size: 8.5pt; color: #777; font-style: italic; margin-top: 4pt; }}
.pagebreak {{ page-break-before: always; }}
"""


def _md_to_html(text: str) -> str:
    text = _sanitize(text)
    return _md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
    )


def _read(name: str) -> str:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def build() -> None:
    today = _dt.date.today().isoformat()

    toc_items = "".join(f"{i+1}.&nbsp;&nbsp;{title}<br/>"
                        for i, (title, _) in enumerate(DOCS))
    cover = f"""
    <div class="cover">
      <h1>Railway Predictive Maintenance</h1>
      <div class="sub">Technical Documentation &mdash; Hackathon MVP</div>
      <div class="sub" style="font-size:11pt;">Anomaly Detection &middot; RUL Estimation &middot;
           Real-time Serving &middot; Fleet Dashboard</div>
      <div class="toc"><b>Contents</b><br/>
        0.&nbsp;&nbsp;System Architecture<br/>{toc_items}</div>
      <div class="meta">Generated {today}</div>
    </div>"""

    arch = f"""
    <div class="pagebreak"></div>
    <h1>System Architecture</h1>
    <p>End-to-end data flow. The <b>offline</b> lane trains and evaluates the
    models; the <b>online</b> lane serves live telemetry in real time. The two
    lanes meet at the Model Store &mdash; models are trained once and loaded
    into the inference engine at startup, so the serving path never re-reads
    disk per event.</p>
    <div class="arch">
      <img src="{_uri(DIAGRAM_PNG)}" />
      <div class="cap">Figure 1 &mdash; Railway PdM system architecture
        (see docs/architecture_diagram.py to regenerate).</div>
    </div>
    """

    sections = []
    for title, fname in DOCS:
        html = _md_to_html(_read(fname))
        sections.append(f'<div class="pagebreak"></div>\n{html}')

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{cover}{arch}{''.join(sections)}</body></html>"""

    with open(OUT_PDF, "wb") as out:
        result = pisa.CreatePDF(full_html, dest=out, encoding="utf-8",
                                link_callback=_link_callback)

    if result.err:
        raise SystemExit(f"PDF generation reported {result.err} error(s).")
    size_kb = os.path.getsize(OUT_PDF) / 1024
    print(f"wrote {OUT_PDF}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    if not os.path.exists(DIAGRAM_PNG):
        print("architecture_diagram.png missing -> generating it first...")
        import architecture_diagram
        architecture_diagram.main()
    build()
