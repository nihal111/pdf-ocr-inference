"""Build a concise vector PDF report from the exported summary.

Requires Python with beautifulsoup4 and system Cairo/Pango libraries.
No browser, server, model, GPU, or network access is used.

    .venv/bin/python -m lab.export_site
    .venv-paddle/bin/python scripts/export_summary_pdf.py
    .venv/bin/python -m lab.export_site
"""
from __future__ import annotations

import argparse
import ctypes as C
import ctypes.util
from html import escape, unescape
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_URL = "https://nihal111.github.io/pdf-ocr-inference/#/doc/results/summary"
PAGE_W, PAGE_H, MARGIN = 595.28, 841.89, 42
WIDTH, BOTTOM = PAGE_W - MARGIN * 2, PAGE_H - 44
INK, MUTED, GREEN = "#1d2939", "#687487", "#147d70"
LINE, PALE, BLUE = "#dce3e8", "#eef7f2", "#edf2fa"
SITE_URL = "https://nihal111.github.io/pdf-ocr-inference/"
# A brief version of docs/pages/theoretical-limits.md; the full page stays on the website.
LIMITS_HTML = """
<h2>Theoretical limits</h2>
<p>The optimized pipeline takes <strong>26.53 s</strong> to process all five PDFs one after
another (20 pages, 136 chunk embeddings, warm models on four L4 GPUs). A best-case model gives
<strong>4.4–5.4 s</strong> for the same serial workload, assuming perfect four-draft
speculative acceptance, peak GPU bandwidth and negligible CPU and transfer overhead. It is an
optimistic bound, not an achievable target.</p>
<table>
<tr><th>Stage, all five PDFs serially</th><th>Measured</th><th>Optimistic minimum</th><th>Stage estimate</th></tr>
<tr><td>Rendering + OCR</td><td>25.76 s</td><td>4.29 s</td><td>5.34 s</td></tr>
<tr><td>Embedding</td><td>0.63 s</td><td>0.08 s</td><td>0.08 s</td></tr>
<tr><td>Preparation, chunking and other overhead</td><td>0.13 s</td><td>0, assumed</td><td>0, assumed</td></tr>
<tr><td><strong>Total</strong></td><td><strong>26.53 s</strong></td><td><strong>4.37 s</strong></td><td><strong>5.43 s</strong></td></tr>
</table>
<ul>
<li><strong>Decoding is bandwidth-bound and sequential.</strong> Each decoder pass reads about
721 MB of BF16 weights, which allows about 416 tokens/s per sequence. Four GPUs can decode
different regions in parallel, but not one region's dependent tokens; the longest Census table
needs 3,252 tokens.</li>
<li><strong>Speculation sets the bound.</strong> With perfect acceptance of four drafts, each
pass advances five tokens. At two tokens per pass the bound rises to 7.7–9.3 s, and to
14–16 s with ordinary decoding.</li>
<li><strong>OCR is where the gap is.</strong> The workload is 24,890 generated tokens across
465 regions. Census (12.36 s measured vs 2.13 s ideal) and W-9 (7.68 s vs 1.68 s) account
for most of the remaining distance.</li>
</ul>
<p><a href="#/doc/pages/theoretical-limits">Full calculation, assumptions and per-PDF breakdown</a></p>
"""


def load_library(name, signatures):
    lib = C.CDLL(ctypes.util.find_library(name))
    for key, (result, args) in signatures.items():
        function = getattr(lib, key)
        function.restype, function.argtypes = result, args
    return lib


P, D, I, S = C.c_void_p, C.c_double, C.c_int, C.c_char_p


class Rectangle(C.Structure):
    _fields_ = [(name, I) for name in ("x", "y", "width", "height")]


CAIRO = load_library("cairo", {
    "cairo_pdf_surface_create": (P, [S, D, D]),
    "cairo_pdf_surface_set_metadata": (None, [P, I, S]),
    "cairo_pdf_surface_add_outline": (I, [P, I, S, S, I]),
    "cairo_surface_finish": (None, [P]),
    "cairo_surface_status": (I, [P]),
    "cairo_surface_destroy": (None, [P]),
    "cairo_create": (P, [P]), "cairo_destroy": (None, [P]),
    "cairo_status": (I, [P]), "cairo_status_to_string": (S, [I]),
    "cairo_set_source_rgb": (None, [P, D, D, D]),
    "cairo_set_line_width": (None, [P, D]),
    "cairo_rectangle": (None, [P, D, D, D, D]),
    "cairo_move_to": (None, [P, D, D]), "cairo_line_to": (None, [P, D, D]),
    "cairo_fill": (None, [P]), "cairo_stroke": (None, [P]),
    "cairo_show_page": (None, [P]),
    "cairo_save": (None, [P]), "cairo_restore": (None, [P]),
    "cairo_translate": (None, [P, D, D]), "cairo_scale": (None, [P, D, D]),
    "cairo_tag_begin": (None, [P, S, S]), "cairo_tag_end": (None, [P, S]),
})
PANGO = load_library("pango-1.0", {
    "pango_layout_set_markup": (None, [P, S, I]),
    "pango_layout_set_width": (None, [P, I]),
    "pango_layout_set_wrap": (None, [P, I]),
    "pango_layout_set_spacing": (None, [P, I]),
    "pango_layout_set_alignment": (None, [P, I]),
    "pango_layout_get_context": (P, [P]),
    "pango_layout_get_text": (S, [P]),
    "pango_layout_index_to_pos": (None, [P, I, C.POINTER(Rectangle)]),
    "pango_font_description_from_string": (P, [S]),
    "pango_font_description_free": (None, [P]),
    "pango_layout_set_font_description": (None, [P, P]),
    "pango_layout_get_size": (None, [P, C.POINTER(I), C.POINTER(I)]),
})
PC = load_library("pangocairo-1.0", {
    "pango_cairo_create_layout": (P, [P]),
    "pango_cairo_show_layout": (None, [P, P]),
    "pango_cairo_context_set_resolution": (None, [P, D]),
})
GOBJECT = load_library("gobject-2.0", {"g_object_unref": (None, [P])})


def markup(node):
    if isinstance(node, NavigableString):
        return escape(re.sub(r"\s+", " ", str(node)))
    if isinstance(node, str):
        return escape(node)
    text = "".join(markup(child) for child in node.children)
    if node.name in ("strong", "b"):
        return f"<b>{text}</b>"
    if node.name in ("em", "i"):
        return f"<i>{text}</i>"
    if node.name == "code":
        return f"<tt>{text}</tt>"
    if node.name == "a":
        return f'<span foreground="{GREEN}">{text}</span>'
    if node.name == "br":
        return "\n"
    if node.name == "mark":
        return f'<span background="{BLUE}">{text}</span>'
    return text


class PDF:
    def __init__(self, path, base_url):
        self.base_url = base_url.rstrip("/") + "/"
        self.surface = CAIRO.cairo_pdf_surface_create(str(path).encode(), PAGE_W, PAGE_H)
        self.ctx = CAIRO.cairo_create(self.surface)
        CAIRO.cairo_pdf_surface_set_metadata(self.surface, 0, b"PDF OCR pipeline optimization - Summary report")
        CAIRO.cairo_pdf_surface_set_metadata(self.surface, 4, b"inference-lab summary report")
        self.page, self.y, self.page_sections = 1, MARGIN, []
        self.pages = []
        self.header("Summary report")

    def color(self, color):
        CAIRO.cairo_set_source_rgb(self.ctx, *(int(color[i:i+2], 16) / 255 for i in (1, 3, 5)))

    def rect(self, x, y, w, h, fill="#ffffff", border=None):
        self.color(fill)
        CAIRO.cairo_rectangle(self.ctx, x, y, w, h)
        CAIRO.cairo_fill(self.ctx)
        if border:
            self.color(border)
            CAIRO.cairo_set_line_width(self.ctx, .65)
            CAIRO.cairo_rectangle(self.ctx, x, y, w, h)
            CAIRO.cairo_stroke(self.ctx)

    def line(self, x1, y1, x2, y2, color=LINE, width=.7):
        self.color(color)
        CAIRO.cairo_set_line_width(self.ctx, width)
        CAIRO.cairo_move_to(self.ctx, x1, y1)
        CAIRO.cairo_line_to(self.ctx, x2, y2)
        CAIRO.cairo_stroke(self.ctx)

    def text(self, text, x, y, width, size=10, color=INK, bold=False, align=0, draw=True, anchors=()):
        layout = PC.pango_cairo_create_layout(self.ctx)
        PC.pango_cairo_context_set_resolution(PANGO.pango_layout_get_context(layout), 72)
        description = PANGO.pango_font_description_from_string(
            f"Noto Sans,DejaVu Sans {'Bold ' if bold else ''}{size}".encode())
        PANGO.pango_layout_set_font_description(layout, description)
        PANGO.pango_font_description_free(description)
        PANGO.pango_layout_set_width(layout, max(1, int(width * 1024)))
        PANGO.pango_layout_set_wrap(layout, 2)
        PANGO.pango_layout_set_spacing(layout, int(size * .25 * 1024))
        PANGO.pango_layout_set_alignment(layout, align)
        PANGO.pango_layout_set_markup(layout, text.encode(), -1)
        w, h = I(), I()
        PANGO.pango_layout_get_size(layout, C.byref(w), C.byref(h))
        if draw:
            self.color(color)
            CAIRO.cairo_move_to(self.ctx, x, y)
            PC.pango_cairo_show_layout(self.ctx, layout)
            plain = PANGO.pango_layout_get_text(layout).decode()
            cursor = 0
            for anchor in anchors:
                label = unescape(re.sub(r"<[^>]+>", "", markup(anchor)))
                start = plain.find(label, cursor)
                if start < 0:
                    continue
                cursor = start + len(label)
                index = len(plain[:start].encode())
                lines = {}
                for char in label:
                    rect = Rectangle()
                    PANGO.pango_layout_index_to_pos(layout, index, C.byref(rect))
                    left, top, rw, rh = [getattr(rect, key)/1024 for key in ("x","y","width","height")]
                    previous = lines.get(top, (left, left+rw, rh))
                    lines[top] = (min(left,previous[0]),max(left+rw,previous[1]),max(rh,previous[2]))
                    index += len(char.encode())
                for top,(left,right,rh) in lines.items():
                    self.annotation(anchor["href"],x+left,y+top,right-left,rh)
        GOBJECT.g_object_unref(layout)
        return h.value / 1024

    def annotation(self, target, x, y, w, h):
        if target.startswith("#/"):
            attributes = f"uri='{self.base_url}{target}'"
        elif target.startswith("#"):
            attributes = f"dest='{target[1:]}'"
        else:
            attributes = f"uri='{target}'"
        CAIRO.cairo_tag_begin(self.ctx,b"Link",
            f"{attributes} rect=[{x} {y} {w} {h}]".encode())
        CAIRO.cairo_tag_end(self.ctx,b"Link")

    def link(self, label, target, x, y, width, size=8):
        if target.startswith("#/"):
            attributes = f"uri='{self.base_url}{target}'"
        elif target.startswith("#"):
            attributes = f"dest='{target[1:]}'"
        else:
            attributes = f"uri='{target}'"
        CAIRO.cairo_tag_begin(self.ctx, b"Link", attributes.encode())
        h = self.text(escape(label), x, y, width, size, GREEN)
        CAIRO.cairo_tag_end(self.ctx, b"Link")
        return h

    def destination(self, ident):
        CAIRO.cairo_tag_begin(self.ctx, b"cairo.dest", f"name='{ident}'".encode())
        CAIRO.cairo_tag_end(self.ctx, b"cairo.dest")

    def header(self, section):
        if self.page > 1:
            self.text("PDF OCR PIPELINE OPTIMIZATION", MARGIN, 20, WIDTH * .65, 7, MUTED)
            self.text(escape(section), MARGIN + WIDTH * .65, 20, WIDTH * .35, 7, MUTED, align=2)
            self.line(MARGIN, 34, PAGE_W - MARGIN, 34)

    def footer(self):
        self.line(MARGIN, PAGE_H - 31, PAGE_W - MARGIN, PAGE_H - 31)
        self.text("Summary report", MARGIN, PAGE_H - 24, WIDTH - 40, 7, MUTED)
        self.text(str(self.page), PAGE_W - MARGIN - 30, PAGE_H - 24, 30, 7, MUTED, align=2)
        self.pages.append({"page": self.page, "sections": self.page_sections, "content_bottom": round(self.y, 2)})

    def new_page(self, section="Summary report"):
        self.footer()
        CAIRO.cairo_show_page(self.ctx)
        self.page += 1
        self.page_sections = []
        self.y = MARGIN
        self.header(section)

    def ensure(self, height):
        if self.y + height > BOTTOM:
            self.new_page()

    def heading(self, title, ident=None, size=16):
        h = self.text(escape(title), MARGIN, self.y, WIDTH, size, bold=True, draw=False)
        self.ensure(h + 65)
        if ident:
            self.destination(ident)
        CAIRO.cairo_pdf_surface_add_outline(self.surface, 0, title.encode(),
            f"page={self.page} pos=[{MARGIN} {self.y}]".encode(), 0)
        self.page_sections.append(title)
        self.y += self.text(escape(title), MARGIN, self.y, WIDTH, size, bold=True) + 10

    def paragraph(self, node, size=9.5, indent=0, bullet=False):
        content = markup(node).strip()
        if not content:
            return
        width = WIDTH - indent
        h = self.text(content, 0, 0, width, size, draw=False)
        self.ensure(h + 10)
        if bullet:
            self.text("•", MARGIN, self.y, 10, size, GREEN)
        anchors = node.find_all("a",href=True) if isinstance(node,Tag) else ()
        self.text(content, MARGIN + indent, self.y, width, size, anchors=anchors)
        self.y += h + 9

    def source_links(self, node):
        links = node.find_all("a", href=True)
        if not links:
            return
        self.ensure(14 * len(links) + 6)
        for a in links:
            self.y += self.link(a.get_text(" ", strip=True), a["href"], MARGIN, self.y, WIDTH, 7.5) + 3
        self.y += 4

    def table(self, table, ratios):
        rows = table.find_all("tr")
        header = rows[0]
        def row_height(row):
            return max(self.text(markup(cell), 0, 0, WIDTH*r-14, 8.1, draw=False)
                       for cell, r in zip(row.find_all(["td","th"], recursive=False), ratios)) + 8
        def paint(row, first=False, index=0):
            height = row_height(row)
            x = MARGIN
            self.rect(x, self.y, WIDTH, height, PALE if first else ("#f7f9fb" if index % 2 == 0 else "#ffffff"))
            for cell, ratio in zip(row.find_all(["td","th"], recursive=False), ratios):
                self.text(markup(cell), x+7, self.y+4, WIDTH*ratio-14, 8.1, bold=first,
                          anchors=cell.find_all("a",href=True))
                x += WIDTH*ratio
            self.y += height
            self.line(MARGIN, self.y, MARGIN+WIDTH, self.y)
        self.ensure(row_height(header) + row_height(rows[1]))
        paint(header, True)
        for i, row in enumerate(rows[1:]):
            if self.y + row_height(row) > BOTTOM:
                self.new_page()
                paint(header, True)
            paint(row, index=i)
        self.y += 12

    def close(self):
        self.footer()
        CAIRO.cairo_show_page(self.ctx)
        CAIRO.cairo_surface_finish(self.surface)
        status = CAIRO.cairo_status(self.ctx) or CAIRO.cairo_surface_status(self.surface)
        CAIRO.cairo_destroy(self.ctx)
        CAIRO.cairo_surface_destroy(self.surface)
        if status:
            raise RuntimeError(CAIRO.cairo_status_to_string(status).decode())


class Diagram:
    """Lay out the webpage's small boxes, grids, chips, and arrows as vectors."""
    def __init__(self, pdf):
        self.pdf = pdf

    def node(self, node, x, y, w, draw=False):
        if isinstance(node, NavigableString):
            text = str(node).strip()
            return self.pdf.text(escape(text), x, y, w, 8, draw=draw) if text else 0
        classes = set(node.get("class", []))
        if "decoding-switch" in classes:
            return 0
        children = [c for c in node.children if isinstance(c, Tag) or str(c).strip()]
        if node.name == "svg":
            horizontal = "horizontal" in classes
            if draw:
                if horizontal:
                    cy, end = y + 8, x + w - 3
                    self.pdf.line(x + 2, cy, end, cy, "#739082", 1)
                    self.pdf.line(end-4, cy-3, end, cy, "#739082", 1)
                    self.pdf.line(end-4, cy+3, end, cy, "#739082", 1)
                else:
                    cx = x+w/2
                    self.pdf.line(cx, y+2, cx, y+14, "#739082", 1)
                    self.pdf.line(cx-3, y+10, cx, y+14, "#739082", 1)
                    self.pdf.line(cx+3, y+10, cx, y+14, "#739082", 1)
            return 17
        if classes & {"change-blocks"}:
            gap, cols = 3, 8
            sw = (w-gap*(cols-1))/cols
            for i,c in enumerate(children):
                if draw:
                    self.pdf.rect(x+(i%cols)*(sw+gap), y+(i//cols)*10, sw, 7,
                                  "#75ad91" if "active" in c.get("class",[]) else "#e5e9ed")
            return ((len(children)+cols-1)//cols)*10
        if "change-precision" in classes:
            span = node.find("span")
            fraction = float(re.search(r"width:(\d+)%", span.get("style","width:100%"))[1])/100
            if draw:
                self.pdf.rect(x,y,w,23,"#f3f5f7",LINE)
                self.pdf.rect(x,y,w*fraction,23,"#d4e8de")
                self.pdf.text(markup(span),x,y+5,w*fraction,8,GREEN,align=1)
            return 23
        if "change-tile" in classes:
            h = 30 + len(node.find_all("span",recursive=False))*7
            if draw:
                self.pdf.rect(x,y,w,h,"#ffffff",LINE)
                self.pdf.text(markup(node.find("strong")),x+5,y+6,w-10,8,bold=True,align=1)
                for i,s in enumerate(node.find_all("span",recursive=False)):
                    self.pdf.rect(x+8,y+26+i*7,w-16,4,
                                  "#75ad91" if "active" in s.get("class",[]) else "#e5e9ed")
            return h
        if classes & {"change-chips","decoding-tokens"}:
            # Wrap chips without shrinking their text.
            cy, cx = y, x
            for child in children:
                label = child.get_text(" ",strip=True)
                cw = min(w, max(24, len(label)*4.4+13))
                if cx > x and cx+cw > x+w:
                    cx,cy=x,cy+25
                if draw:
                    c=set(child.get("class",[]))
                    fill,color=(BLUE,"#3d536f")
                    if c & {"active","accepted"}: fill,color="#e5f3eb","#286343"
                    if "waiting" in c: fill,color="#fff1db","#856127"
                    if "rejected" in c: fill,color="#fce9e7","#a34132"
                    if "discarded" in c: fill,color="#f1f2f3","#8d959c"
                    self.pdf.rect(cx,cy,cw,21,fill,LINE)
                    self.pdf.text(escape(label),cx+3,cy+4,cw-6,8,color,align=1)
                cx += cw+4
            return cy-y+23
        grid = classes & {"change-comparison","change-gpus","change-capacity-state",
                           "decoding-lanes","decoding-context","change-tile-group"}
        if classes & {"change-gpus","change-capacity-state"}:
            cw=(w-10)/2
            cy=y
            for start in range(0,len(children),2):
                heights=[self.node(child,x+i*(cw+10),cy,cw,draw)
                         for i,child in enumerate(children[start:start+2])]
                cy+=max(heights)+8
            return cy-y-8
        if grid or "decoding-pass" in classes or "decoding-feedback" in classes:
            if "decoding-pass" in classes: widths=[w*.57,18,w*.43-26]
            elif "decoding-feedback" in classes: widths=[18,w-26]
            else: widths=[(w-10*(len(children)-1))/len(children)]*len(children)
            heights=[]
            for child, cw in zip(children,widths):
                heights.append(self.node(child,x,y,cw,draw))
                x += cw + (4 if classes & {"decoding-pass","decoding-feedback"} else 10)
            return max(heights,default=0)
        boxes = classes & {"change-lane","change-node","change-gpu","change-worker",
                            "decoding-lane","decoding-model","decoding-history",
                            "decoding-proposer","decoding-image","decoding-output"}
        if boxes:
            pad=9
            h=self.flow(children,x+pad,y+pad,w-2*pad,False)+2*pad
            if draw:
                fill="#ffffff"
                if classes & {"after","model","active","selected","speculative","decoding-model"}:
                    fill=PALE
                if "idle" in classes: fill="#f4f5f6"
                self.pdf.rect(x,y,w,h,fill,LINE)
                self.flow(children,x+pad,y+pad,w-2*pad,True)
            return h
        if node.name in ("strong","h3","h4","small","span","p","em","figcaption","mark") or "decoding-input-label" in classes:
            size = 9 if node.name in ("h3","h4") else 7.4 if node.name in ("small","figcaption") else 8
            bold = node.name in ("strong","h3","h4")
            color = GREEN if node.name in ("h3","h4") else MUTED if node.name in ("small","figcaption") or "decoding-input-label" in classes else INK
            return self.pdf.text(markup(node),x,y,w,size,color,bold,draw=draw)
        return self.flow(children,x,y,w,draw)

    def flow(self, children, x,y,w,draw):
        cy=y
        for child in children:
            h=self.node(child,x,cy,w,draw)
            if h: cy += h+5
        return max(0,cy-y-5)

    def render(self, html, max_height=None):
        soup=BeautifulSoup(html,"html.parser")
        h=self.node(soup,MARGIN,self.pdf.y,WIDTH)
        scale=min(1,max_height/h) if max_height else 1
        if scale < .76:
            raise ValueError(f"Diagram would be too small to read: scale={scale:.2f}")
        self.pdf.ensure(h*scale+10)
        CAIRO.cairo_save(self.pdf.ctx)
        CAIRO.cairo_translate(self.pdf.ctx,MARGIN,self.pdf.y)
        CAIRO.cairo_scale(self.pdf.ctx,scale,scale)
        self.node(soup,0,0,WIDTH,True)
        CAIRO.cairo_restore(self.pdf.ctx)
        self.pdf.y += h*scale+12
        return scale


def table_rows(table):
    return [[c.get_text(" ",strip=True) for c in row.find_all("td",recursive=False)]
            for row in table.select("tbody > tr")]


def pipeline(pdf, table):
    rows=table_rows(table)
    gap=13
    w=(WIDTH-gap*3)/4
    height=max(
        pdf.text(f"0{i+1}   <b>{escape(stage)}</b>",0,0,w-16,10,draw=False)
        +pdf.text(escape(transformation),0,0,w-16,8.2,draw=False)
        +pdf.text(escape(technology),0,0,w-16,7.6,draw=False)+49
        for i,(stage,transformation,technology) in enumerate(rows))
    pdf.ensure(height+10)
    for i,(stage,transformation,technology) in enumerate(rows):
        x=MARGIN+i*(w+gap)
        pdf.rect(x,pdf.y,w,height,PALE if i==1 else "#f7f9fb",LINE)
        cy=pdf.y+10
        cy += pdf.text(f"0{i+1}   <b>{escape(stage)}</b>",x+8,cy,w-16,10,GREEN)+10
        cy += pdf.text(escape(transformation),x+8,cy,w-16,8.2)+10
        pdf.line(x+8,cy,x+w-8,cy)
        pdf.text(escape(technology),x+8,cy+9,w-16,7.6,GREEN)
        if i<3:
            pdf.line(x+w+2,pdf.y+height/2,x+w+gap-2,pdf.y+height/2,GREEN)
            pdf.line(x+w+gap-5,pdf.y+height/2-3,x+w+gap-2,pdf.y+height/2,GREEN)
    pdf.y += height+16


def build(root, output, base_url):
    record=json.loads((root/"site/site-data/docs/results/summary.json").read_text())
    soup=BeautifulSoup(record["html"],"html.parser")
    tables=soup.find_all("table")
    pipeline_table=next(t for t in tables if t.find("th").get_text()=="Stage")
    speculation_table=next(t for t in tables if t.find("th").get_text()=="Decoding example")
    changes_table=next(t for t in tables if t.find("th").get_text()=="Change")
    spec_intro=speculation_table.find_previous_sibling("p")
    spec_conclusion=speculation_table.find_next_sibling("p")
    spec_intro.decompose()
    speculation_table.decompose()
    spec_conclusion.decompose()
    changes_table.find_previous_sibling("p").string=(
        "Each row compares a retained change with its recorded control, in experiment order.")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(f".{output.name}.tmp")
    pdf=PDF(temporary,base_url)
    pdf.text("PERFORMANCE REPORT",MARGIN,pdf.y,WIDTH,8,GREEN,bold=True)
    pdf.y+=20
    pdf.y+=pdf.text("PDF OCR pipeline optimization",MARGIN,pdf.y,WIDTH,23,bold=True)+9
    pdf.y+=pdf.text("Summary",MARGIN,pdf.y,WIDTH,10,MUTED)+8
    pdf.y+=pdf.link(f"Check out {SITE_URL} for a more interactive view",
                    SITE_URL,MARGIN,pdf.y,WIDTH,9)+18

    def block(node):
        if isinstance(node,NavigableString): return
        if node.name=="h2":
            if node.get_text() in ("Quality problems before optimization", "Changes that paid off"):
                pdf.new_page()
            pdf.y+=7
            pdf.heading(node.get_text())
        elif node is pipeline_table:
            pipeline(pdf,node)
        elif node.name=="table":
            ratios=[.59,.13,.13,.15] if node is changes_table else [.34,.24,.26,.16] if len(node.find_all("th"))==4 else [.35,.65]
            pdf.table(node,ratios)
        elif node.name=="p":
            pdf.paragraph(node)
        elif node.name in ("ul","ol"):
            for li in node.find_all("li",recursive=False):
                children=list(li.children)
                if li.find("p",recursive=False):
                    for child in children:
                        if isinstance(child,Tag) and child.name=="p":
                            pdf.paragraph(child,indent=12,bullet=(child is li.find("p",recursive=False)))
                else: pdf.paragraph(li,indent=12,bullet=True)
    for child in list(soup.children): block(child)
    limits=BeautifulSoup(LIMITS_HTML,"html.parser")
    limits_table=limits.find("table")
    for child in list(limits.children):
        if child is limits_table: pdf.table(child,[.46,.18,.18,.18])
        else: block(child)
    pdf.y+=5
    pdf.ensure(24)
    pdf.y+=pdf.link(f"For more details, please open {SUMMARY_URL}",
                    SUMMARY_URL,MARGIN,pdf.y,WIDTH,8.5)+5
    pdf.close()
    if any(page["content_bottom"] > BOTTOM for page in pdf.pages):
        raise ValueError("Report content overlaps a page footer")
    if temporary.stat().st_size > 1_000_000:
        raise ValueError("Report exceeds the website's per-file size limit")
    temporary.replace(output)
    metadata={"title":"PDF OCR pipeline optimization","pages":pdf.pages,
              "appendix_sections":0,"details_url":SUMMARY_URL,
              "source_summary":"docs/results/summary.md","pdf_bytes":output.stat().st_size,
              "base_url":base_url}
    audit=root/".site/summary-report.json"
    audit.parent.mkdir(parents=True,exist_ok=True)
    audit.write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"pdf":str(output),"pages":len(pdf.pages),"bytes":output.stat().st_size,
                      "appendix_sections":0}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=ROOT/"web/downloads/summary-report.pdf")
    parser.add_argument("--base-url",default="https://nihal111.github.io/pdf-ocr-inference/")
    args=parser.parse_args()
    build(args.root,args.output,args.base_url)
