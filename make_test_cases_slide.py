"""1-page slide: Generating Test Cases — Systematic Coverage via Design of Experiments.

Concept first: can't test everything → design the experiment.
Three-panel arc: Generate → Expand → Attribute.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- Colors --------------------------------------------------------------
NAVY   = RGBColor(0x14, 0x2B, 0x4D)
DARK   = RGBColor(0x22, 0x26, 0x2E)
BODY   = RGBColor(0x3E, 0x44, 0x50)
MID    = RGBColor(0x6C, 0x72, 0x80)
FAINT  = RGBColor(0xD3, 0xD7, 0xDE)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

STEP1  = RGBColor(0x1F, 0x6F, 0xB5)   # Generate  — blue
STEP2  = RGBColor(0xC9, 0x56, 0x2D)   # Design    — orange
STEP3  = RGBColor(0x2E, 0x8B, 0x57)   # Analyze   — green

STEP1_BG = RGBColor(0xE6, 0xF0, 0xFA)
STEP2_BG = RGBColor(0xFB, 0xEB, 0xDD)
STEP3_BG = RGBColor(0xE7, 0xF4, 0xEC)

COG_TINT    = RGBColor(0xE6, 0xF0, 0xFA)   # cognitive pills
STRESS_TINT = RGBColor(0xFB, 0xEB, 0xDD)   # stress pills

MONO = "Consolas"
SANS = "Calibri"


def txt(slide, left, top, width, height, runs,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        margin=(0.02, 0.02, 0.02, 0.02)):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left   = Inches(margin[0])
    tf.margin_right  = Inches(margin[1])
    tf.margin_top    = Inches(margin[2])
    tf.margin_bottom = Inches(margin[3])
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for text, size, bold, italic, color, font in para:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = color
            r.font.name = font
    return tb


def box(slide, left, top, width, height, fill, line=None, rounded=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        left, top, width, height,
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    return shape


def arrow(slide, left, top, width, height, color):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    return s


def pill(slide, left, top, width, height, fill, text_color, label, font_size=9.5):
    box(slide, left, top, width, height, fill, rounded=True)
    txt(slide, left, top, width, height, [[
            (label, font_size, True, False, text_color, SANS),
        ]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# --- Slide ---------------------------------------------------------------
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Navy stripe
box(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.10), NAVY)

# Title
txt(slide, Inches(0.5), Inches(0.22), Inches(12.4), Inches(0.55), [[
    ("Generating Test Cases", 30, True, False, NAVY, SANS),
    ("   —   systematic coverage through design of experiments",
     17, False, False, MID, SANS),
]])
txt(slide, Inches(0.5), Inches(0.78), Inches(12.4), Inches(0.28), [[
    ("Essay 3  ·  Agus Sudjianto  &  Aldo Lau", 11, False, True, MID, SANS),
]])

# One-line framing
txt(slide, Inches(0.5), Inches(1.18), Inches(12.4), Inches(0.40), [[
    ("807 facts to test.  40 presentation factors.  ",
     14, True, False, DARK, SANS),
    ("The combinations run past 12 quintillion — so we don't sample randomly, we ",
     14, False, False, BODY, SANS),
    ("design the experiment", 14, True, True, STEP2, SANS),
    (".", 14, False, False, BODY, SANS),
]])

# -- Three-panel concept flow ---------------------------------------------
t_top   = Inches(1.80)
t_h     = Inches(3.35)
panel_w = Inches(3.85)
gap     = Inches(0.45)
start_l = Inches(0.5 + (12.4 - (3 * 3.85 + 2 * 0.45)) / 2.0)

def make_panel(left, accent, accent_bg, title, tagline, body_runs, footer):
    box(slide, left, t_top, panel_w, t_h, fill=accent_bg, line=accent, rounded=True)
    # header
    box(slide, left, t_top, panel_w, Inches(0.45), fill=accent, rounded=True)
    box(slide, left, t_top + Inches(0.33), panel_w, Inches(0.12), fill=accent)
    txt(slide, left + Inches(0.15), t_top + Inches(0.04),
        panel_w - Inches(0.3), Inches(0.38), [[
            (title, 16, True, False, WHITE, SANS),
            ("   " + tagline, 11, False, True, WHITE, SANS),
        ]], anchor=MSO_ANCHOR.MIDDLE)
    # body card
    box(slide, left + Inches(0.25), t_top + Inches(0.65),
        panel_w - Inches(0.5), Inches(2.15), fill=WHITE, line=FAINT, rounded=True)
    txt(slide, left + Inches(0.40), t_top + Inches(0.80),
        panel_w - Inches(0.8), Inches(2.0), body_runs)
    # footer outcome
    txt(slide, left + Inches(0.20), t_top + t_h - Inches(0.42),
        panel_w - Inches(0.4), Inches(0.35), [[
            (footer, 11.5, True, True, accent, SANS),
        ]], align=PP_ALIGN.CENTER)


# Panel 1 — Generate
l1 = start_l
make_panel(
    l1, STEP1, STEP1_BG,
    "Generate", "from the ground truth",
    [
        [("Thirteen question generators.",
          12.5, True, False, DARK, SANS)],
        [("Each maps to one failure mode.",
          11.5, False, True, BODY, SANS)],
        [(" ", 6, False, False, DARK, SANS)],
        [("·  8 cognitive  ",  11.5, True, False, STEP1, SANS),
         ("— probe reasoning over the graph",
          11,  False, False, BODY, SANS)],
        [("·  5 stress  ",      11.5, True, False, STEP1, SANS),
         ("— probe behavior under pressure",
          11,  False, False, BODY, SANS)],
        [(" ", 6, False, False, DARK, SANS)],
        [("Every answer is verified by ",
          11, False, False, BODY, SANS),
         ("the store itself", 11, True, False, DARK, SANS),
         ("  —  no LLM judgment in generation.",
          11, False, False, BODY, SANS)],
    ],
    "~130 balanced base questions",
)

# arrow 1
arrow(slide, l1 + panel_w + Inches(0.05), t_top + Inches(1.45),
      Inches(0.35), Inches(0.40), STEP2)

# Panel 2 — Design
l2 = l1 + panel_w + gap
make_panel(
    l2, STEP2, STEP2_BG,
    "Design", "space-filling DoE",
    [
        [("Each base question × design matrix.",
          12.5, True, False, DARK, SANS)],
        [("Vary the ", 11.5, False, True, BODY, SANS),
         ("presentation", 11.5, True,  True, DARK, SANS),
         (", not the content.",
          11.5, False, True, BODY, SANS)],
        [(" ", 6, False, False, DARK, SANS)],
        [("·  Sobol: every factor level sampled",
          11, False, False, BODY, SANS)],
        [("·  Ground truth preserved per row",
          11, False, False, BODY, SANS)],
        [("·  Seed-pinned and reproducible",
          11, False, False, BODY, SANS)],
        [(" ", 6, False, False, DARK, SANS)],
        [("Factors: ", 11, True, False, STEP2, SANS),
         ("clarity, aliasing, cue, noise, context, conflict, tool, …",
          11, False, True, BODY, SANS)],
    ],
    "130  →  520 test cases  (4× expansion)",
)

# arrow 2
arrow(slide, l2 + panel_w + Inches(0.05), t_top + Inches(1.45),
      Inches(0.35), Inches(0.40), STEP3)

# Panel 3 — Attribute
l3 = l2 + panel_w + gap
make_panel(
    l3, STEP3, STEP3_BG,
    "Attribute", "logistic regression",
    [
        [("Binary outcomes.  Factor-level attribution.",
          12.5, True, False, DARK, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("·  ",         11, True,  False, STEP3, SANS),
         ("which factor level hurt accuracy?",
          11, False, False, BODY, SANS)],
        [("·  ",         11, True,  False, STEP3, SANS),
         ("by how much?  odds ratios · Wilson CIs",
          11, False, False, BODY, SANS)],
        [("·  ",         11, True,  False, STEP3, SANS),
         ("where do factors compound?  interactions",
          11, False, False, BODY, SANS)],
        [(" ", 6, False, False, DARK, SANS)],
        [("Not ", 11, False, False, BODY, SANS),
         ("\"64% correct\"", 11, True, True, MID, MONO),
         (".  ", 11, False, False, BODY, SANS),
         ("Instead: ", 11, False, False, BODY, SANS),
         ("\"Misleading wording cuts odds to 0.68×\"",
          11, True, False, STEP3, SANS),
         (".", 11, False, False, BODY, SANS)],
    ],
    "actionable  —  per-factor, with confidence",
)

# -- Thirteen generators row ----------------------------------------------
gen_head_top = Inches(5.30)
txt(slide, Inches(0.5), gen_head_top, Inches(12.4), Inches(0.3), [[
    ("Thirteen generators:",   12, True, False, NAVY, SANS),
    ("  8 cognitive  +  5 stress  —  one per targeted failure mode.",
     11.5, False, True, MID, SANS),
]])

# Cognitive row
cog_pills = [
    "plausibility", "contradiction", "path check", "exact recall",
    "link",         "multi-hop",     "conditional", "procedural",
]
stress_pills = [
    "boundary", "adversarial", "noise", "tool failure", "policy trap",
]

pill_top_1 = Inches(5.65)
pill_h     = Inches(0.28)
cog_slot_w = (12.4 - 0.10 * 7) / 8  # 8 pills across 12.4" with 0.1 gap
cog_slot_w_in = Inches(cog_slot_w)
for i, p in enumerate(cog_pills):
    left = Inches(0.5) + i * (cog_slot_w_in + Inches(0.10))
    pill(slide, left, pill_top_1, cog_slot_w_in, pill_h,
         COG_TINT, STEP1, p, font_size=10)

# Stress row
pill_top_2 = pill_top_1 + pill_h + Inches(0.12)
stress_slot_w = (12.4 - 0.10 * 4) / 5
stress_slot_w_in = Inches(stress_slot_w)
for i, p in enumerate(stress_pills):
    left = Inches(0.5) + i * (stress_slot_w_in + Inches(0.10))
    pill(slide, left, pill_top_2, stress_slot_w_in, pill_h,
         STRESS_TINT, STEP2, p, font_size=10)

# -- Result bar -----------------------------------------------------------
bar_top = Inches(6.70)
box(slide, Inches(0.5), bar_top, Inches(12.4), Inches(0.65), NAVY, rounded=True)
txt(slide, Inches(0.75), bar_top + Inches(0.08), Inches(11.9), Inches(0.50), [
    [
        ("Deterministic pipeline.  ",
         11.5, True, False, WHITE, SANS),
        ("Same seed, same 520 test cases, same verdicts every run  —  regression testing becomes a diff, not a re-experiment.",
         11.5, False, False, WHITE, SANS),
    ],
    [
        ("Finding on the running example: ",
         10.5, False, True, WHITE, SANS),
        ("Misleading wording fails 47.2% vs 22.1% for Clear (non-overlapping 95% CIs)  —  hardening target, identified.",
         10.5, False, True, WHITE, SANS),
    ],
])

out = "/home/asudjianto/jupyterlab/kg-memory/test_cases_slide.pptx"
prs.save(out)
print(f"Wrote: {out}")
