# -*- coding: utf-8 -*-
"""Graphic abstract, revised version (single editable PPTX slide).

Follows the revised manuscript: evaluation audit, model, reachable performance
and limits under LDO.
Numbers come from the revised manuscript: 233 drug-name groups / 198,606 pairs /
941 cell lines; R2 0.4134 after window clamping, Pearson 0.6468 (drug-block
bootstrap, B=1000, 95% CI 0.298-0.494); R2_level 0.4774, R2_within 0.2559;
71.10% of label variance lies between drugs; 62.9% of squared error is a
per-drug constant offset; 44/111 targets fall outside the 2048-gene panel;
cross-seed prediction correlation 0.9705, attention Spearman 0.105, CHEK1 7->811.
"""
import os
import sys

VENDOR = r'd:\ProgramData\my_model\_vendor'
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

OUT_PPTX = r'd:\ProgramData\my_model\submit\Graphic_Abstract_20261110.pptx'

W, H = 13.333, 7.5
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x5A, 0x5A, 0x5A)
LIGHT = RGBColor(0x8E, 0x8E, 0x8E)
RULE = RGBColor(0xD9, 0xD9, 0xD9)
ACCENT = RGBColor(0xB2, 0x18, 0x2B)
ACCENT_L = RGBColor(0xF4, 0xE4, 0xE6)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
BLUE_L = RGBColor(0xDF, 0xEA, 0xF3)
GREY_L = RGBColor(0xF1, 0xF1, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = 'Arial'

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def no_shadow(sh):
    try:
        sh.shadow.inherit = False
    except Exception:
        pass


def rect(x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         adj=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    no_shadow(sh)
    if adj is not None:
        try:
            sh.adjustments[0] = adj
        except Exception:
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.text_frame.word_wrap = True
    return sh


def text(x, y, w, h, parts, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=2, line_spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = parts if isinstance(parts[0], list) else [parts]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        para.line_spacing = line_spacing
        for t, size, bold, color in p:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = FONT
    return tb


def chip(x, y, w, h, label, fill, color=WHITE, size=9.5, bold=True):
    sh = rect(x, y, w, h, fill=fill, line=None, adj=0.22)
    tf = sh.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT
    return sh


def down_arrow(cx, y, h=0.18, color=LIGHT):
    sh = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx - 0.045),
                                Inches(y), Inches(0.09), Inches(h))
    no_shadow(sh)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


# ================= title =================
text(0.40, 0.24, 12.5, 0.42,
     [("Bidirectional cross-attention for leave-drug-out anticancer drug response prediction",
       19, True, INK)])
text(0.40, 0.70, 12.5, 0.28,
     [("GDSC2  \u00b7  941 cell lines  \u00b7  233 drug groups  \u00b7  198,606 drug\u2013cell "
       "pairs  \u00b7  five-fold leave-drug-out, two seeds", 9.5, False, GREY)])
rect(0.40, 1.04, 12.54, 0.012, fill=RULE, shape=MSO_SHAPE.RECTANGLE)

COLX = [0.40, 4.62, 8.84]
COLW = 3.95
CHIP_Y = 1.20

# ================= column 1: evaluation question =================
chip(COLX[0], CHIP_Y, COLW, 0.30, '1   The evaluation question', INK)
text(COLX[0], 1.62, COLW, 0.80,
     [[("Random and leave-cell-line-out splits keep the test drug in training.",
        9, False, INK)],
      [("LDO removes the compound itself \u2014 the setting a new drug actually faces.",
        9, False, INK)]])

rect(COLX[0], 2.42, COLW, 1.26, fill=GREY_L, line=None, adj=0.06)
text(COLX[0] + 0.16, 2.52, 1.20, 0.40, [("71.1%", 21, True, ACCENT)])
text(COLX[0] + 0.16, 2.88, COLW - 0.32, 0.22,
     [("of label variance lies between drugs, not within them", 8, False, GREY)])
text(COLX[0] + 0.16, 3.14, 1.20, 0.36, [("62.9%", 21, True, INK)])
text(COLX[0] + 0.16, 3.50, COLW - 0.32, 0.22,
     [("of squared error is a per-drug constant offset (median 1.05 ln)",
       8, False, GREY)])

# label-variance split bar
text(COLX[0], 3.78, COLW, 0.20, [("label variance", 7.5, False, GREY)])
bx1, bw1 = COLX[0], COLW
rect(bx1, 3.98, bw1 * 0.711, 0.20, fill=INK, line=None, shape=MSO_SHAPE.RECTANGLE)
rect(bx1 + bw1 * 0.711, 3.98, bw1 * 0.289, 0.20, fill=RGBColor(0xC8, 0xC8, 0xC8),
     line=None, shape=MSO_SHAPE.RECTANGLE)
text(bx1 + 0.06, 4.00, 1.60, 0.18, [("between drugs  71.1%", 7.5, True, WHITE)])
text(bx1 + bw1 * 0.711 + 0.06, 4.00, 1.10, 0.18,
     [("within  28.9%", 7.5, True, INK)])

text(COLX[0], 4.32, COLW, 0.66,
     [[("Reported, not patched:", 8.5, True, INK),
       (" the drug-grouping key merges 9 compounds (14,726 rows); LDO is run on drug "
        "name, not drug identity.", 8.5, False, GREY)]])
text(COLX[0], 5.02, COLW, 0.66,
     [[("Column floor:", 8.5, True, INK),
       (" a cell-line-mean baseline already reaches R\u00b2 0.3184, so every claim is "
        "measured against it.", 8.5, False, GREY)]])
text(COLX[0], 5.72, COLW, 0.66,
     [[("Interval:", 8.5, True, INK),
       (" drug-block bootstrap, B = 1000, gives 0.298\u20130.494 for pooled R\u00b2 and "
        "0.559\u20130.710 for Pearson.", 8.5, False, GREY)]])

# ================= column 2: model =================
chip(COLX[1], CHIP_Y, COLW, 0.30, '2   Model', INK)
y = 1.62
rect(COLX[1], y, COLW, 0.48, fill=BLUE_L, line=BLUE, lw=0.5, adj=0.10)
text(COLX[1] + 0.14, y + 0.10, COLW - 0.28, 0.32,
     [[("Gene expression + Geneformer", 8.6, True, BLUE),
       ("  \u2192  2048-gene panel", 8.6, False, INK)]])
y2 = y + 0.58
rect(COLX[1], y2, COLW, 0.48, fill=BLUE_L, line=BLUE, lw=0.5, adj=0.10)
text(COLX[1] + 0.14, y2 + 0.10, COLW - 0.28, 0.32,
     [[("SMILES", 8.6, True, BLUE),
       ("  \u2192  Morgan 2048 bits + FG-BERT atom embeddings", 8.6, False, INK)]])
down_arrow(COLX[1] + COLW / 2, y2 + 0.52, 0.20)

y3 = y2 + 0.78
rect(COLX[1] + 0.26, y3, COLW - 0.52, 0.44, fill=INK, line=None, adj=0.14)
text(COLX[1] + 0.26, y3 + 0.11, COLW - 0.52, 0.24,
     [("Bidirectional cross-attention  \u00b7  8 heads", 8.6, True, WHITE)],
     align=PP_ALIGN.CENTER)
down_arrow(COLX[1] + COLW / 2, y3 + 0.48, 0.20)

y4 = y3 + 0.74
for i, lab in enumerate(('Drug Global', 'Gene Pooled', 'Global Gene')):
    rect(COLX[1] + i * 1.30, y4, 1.18, 0.34, fill=WHITE, line=GREY, lw=0.5, adj=0.18)
    text(COLX[1] + i * 1.30, y4 + 0.085, 1.18, 0.20, [(lab, 7.5, False, INK)],
         align=PP_ALIGN.CENTER)
down_arrow(COLX[1] + COLW / 2, y4 + 0.38, 0.20)

y5 = y4 + 0.64
rect(COLX[1] + 0.26, y5, COLW - 0.52, 0.40, fill=GREY_L, line=None, adj=0.14)
text(COLX[1] + 0.26, y5 + 0.10, COLW - 0.52, 0.24,
     [("content-aware gate  \u2192  MLP  \u2192  ln(IC\u2085\u2080)", 8.6, True, INK)],
     align=PP_ALIGN.CENTER)

y6 = y5 + 0.54
rect(COLX[1], y6, COLW, 0.96, fill=ACCENT_L, line=ACCENT, lw=0.5, adj=0.06)
text(COLX[1] + 0.14, y6 + 0.10, COLW - 0.28, 0.80,
     [[("Input ceiling.", 8.6, True, ACCENT),
       ("  44 of 111 drug targets never appear in the 2048-gene panel (median presence "
        "0.32%), so attention cannot carry target biology from a position that is not "
        "there.", 8.6, False, INK)]])
text(COLX[1], y6 + 1.06, COLW, 0.44,
     [[("Gate weights are learned per sample; removing any one modality lowers "
        "R\u00b2 in most folds.", 8.5, False, GREY)]])

# ================= column 3: results and limits =================
chip(COLX[2], CHIP_Y, COLW, 0.30, '3   What it reaches', INK)
text(COLX[2], 1.58, COLW, 0.22,
     [("Pooled R\u00b2 after per-drug clamp, same folds and same labels", 7.8, False, GREY)])

rows = [('BCA-HFP', 0.4134, ACCENT),
        ('tCNNS', 0.3845, LIGHT),
        ('PaccMann', 0.3506, LIGHT),
        ('TransCDR', 0.3375, LIGHT),
        ('cell-line mean', 0.3184, BLUE),
        ('GraTransDRP', 0.2976, LIGHT),
        ('GraphDRP', 0.2067, LIGHT)]
SCALE = 0.55
bx, bw = COLX[2] + 1.10, 2.02
lx = bx + bw + 0.08
ry = 1.86
for name, v, col in rows:
    text(COLX[2], ry + 0.02, 1.04, 0.20, [(name, 7.6, name == 'BCA-HFP', INK)],
         align=PP_ALIGN.RIGHT)
    rect(bx, ry + 0.02, bw * v / SCALE, 0.13, fill=col, line=None,
         shape=MSO_SHAPE.RECTANGLE)
    text(lx, ry + 0.00, 0.56, 0.20, [('%.4f' % v, 7.6, name == 'BCA-HFP', INK)])
    if name == 'BCA-HFP':
        x_lo, x_hi = bx + bw * 0.298 / SCALE, bx + bw * 0.494 / SCALE
        rect(x_lo, ry + 0.175, x_hi - x_lo, 0.028, fill=INK, line=None,
             shape=MSO_SHAPE.RECTANGLE)
        for xx in (x_lo, x_hi):
            rect(xx - 0.011, ry + 0.145, 0.022, 0.088, fill=INK, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
    ry += 0.268
text(COLX[2], ry + 0.02, COLW, 0.22,
     [("black bar = 95% CI 0.298\u20130.494  \u00b7  Pearson 0.6468  \u00b7  pre-clamp 0.2301",
       7.4, False, GREY)])

ry += 0.44
text(COLX[2], ry, COLW, 0.22,
     [("The strength is between drugs, not within:", 8.6, True, INK)])
text(COLX[2], ry + 0.22, COLW, 0.24,
     [("R\u00b2_level 0.4774   vs   R\u00b2_within 0.2559", 8.6, False, INK)])
text(COLX[2], ry + 0.48, COLW, 0.44,
     [("Ranking drugs by potency transfers to unseen compounds; ranking cell lines "
       "for a given drug transfers much less.", 8.5, False, GREY)])

ry += 1.06
rect(COLX[2], ry, COLW, 0.92, fill=ACCENT_L, line=ACCENT, lw=0.5, adj=0.07)
text(COLX[2] + 0.14, ry + 0.10, COLW - 0.28, 0.74,
     [[("Attention is not a mechanism.", 8.6, True, ACCENT),
       (" Predictions reproduce across seeds (r = 0.9705); attention does not \u2014 "
        "per-gene Spearman 0.105, CHEK1 rank 7 \u2192 811.", 8.6, False, INK)]])

# ================= bottom line =================
rect(0.40, 6.62, 12.54, 0.012, fill=RULE, shape=MSO_SHAPE.RECTANGLE)
text(0.40, 6.74, 12.54, 0.44,
     [[("Under leave-drug-out the model ranks drugs far better than it ranks cell lines "
        "for a given drug, and its attention weights describe input positions, "
        "not mechanism.", 9.5, True, INK)]])

prs.save(OUT_PPTX)
print('saved', OUT_PPTX)
