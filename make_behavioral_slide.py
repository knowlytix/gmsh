"""1-page slide: Behavioral Testing.

Grounded in gmsh_testing_ext:
  - Ch 16:  Hallucination Detection via Geodesic Distance
  - App. G: Calibration Toolkit - Angular Encoding, ECE, MCE
  - Ch 17:  Stateful and Multi-Turn Testing (teach-delay-recall)

Three-panel arc:
  Hallucination bands   ->   Angular calibration   ->   Teach-delay-recall.
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

RED    = RGBColor(0xB2, 0x3A, 0x2F)   # hallucination (is it grounded?)
BLUE   = RGBColor(0x1F, 0x6F, 0xB5)   # calibration (is confidence honest?)
GREEN  = RGBColor(0x2E, 0x8B, 0x57)   # multi-turn (does it remember?)

RED_BG    = RGBColor(0xFB, 0xE9, 0xE7)
BLUE_BG   = RGBColor(0xE6, 0xF0, 0xFA)
GREEN_BG  = RGBColor(0xE7, 0xF4, 0xEC)

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


# --- Slide ---------------------------------------------------------------
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Navy stripe
box(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.10), NAVY)

# Title
txt(slide, Inches(0.5), Inches(0.22), Inches(12.4), Inches(0.55), [[
    ("Behavioral Testing", 30, True, False, NAVY, SANS),
    ("   —   hallucination, calibration, multi-turn memory",
     17, False, False, MID, SANS),
]])
txt(slide, Inches(0.5), Inches(0.78), Inches(12.4), Inches(0.28), [[
    ("Essay 7  ·  Agus Sudjianto  &  Aldo Lau", 11, False, True, MID, SANS),
]])

# One-line framing
txt(slide, Inches(0.5), Inches(1.18), Inches(12.4), Inches(0.42), [[
    ("Some failure modes need their own probes.  ",
     13.5, True, False, DARK, SANS),
    ("Three specialised protocols, all reading the ",
     13.5, False, False, BODY, SANS),
    ("same calibrated manifold",
     13.5, True, True, BLUE, SANS),
    (".",
     13.5, False, False, BODY, SANS),
]])

# -- Three-panel concept arc ----------------------------------------------
t_top   = Inches(1.78)
t_h     = Inches(3.75)
panel_w = Inches(3.85)
gap     = Inches(0.45)
start_l = Inches(0.5 + (12.4 - (3 * 3.85 + 2 * 0.45)) / 2.0)


def make_panel(left, accent, accent_bg, title, tagline, body_runs, footer_runs):
    box(slide, left, t_top, panel_w, t_h, fill=accent_bg, line=accent, rounded=True)
    box(slide, left, t_top, panel_w, Inches(0.45), fill=accent, rounded=True)
    box(slide, left, t_top + Inches(0.33), panel_w, Inches(0.12), fill=accent)
    txt(slide, left + Inches(0.15), t_top + Inches(0.04),
        panel_w - Inches(0.3), Inches(0.38), [[
            (title, 16, True, False, WHITE, SANS),
            ("   " + tagline, 11, False, True, WHITE, SANS),
        ]], anchor=MSO_ANCHOR.MIDDLE)
    box(slide, left + Inches(0.25), t_top + Inches(0.65),
        panel_w - Inches(0.5), Inches(2.55), fill=WHITE, line=FAINT, rounded=True)
    txt(slide, left + Inches(0.38), t_top + Inches(0.78),
        panel_w - Inches(0.76), Inches(2.42), body_runs)
    txt(slide, left + Inches(0.20), t_top + t_h - Inches(0.42),
        panel_w - Inches(0.4), Inches(0.35), footer_runs,
        align=PP_ALIGN.CENTER)


# ---------- Panel 1: Hallucination (Ch 16) -------------------------------
l1 = start_l
make_panel(
    l1, RED, RED_BG,
    "Hallucination",
    "is the claim grounded?",
    [
        [("Score each claim's ",
          11.5, False, False, BODY, SANS),
         ("geodesic distance",
          11.5, True, False, DARK, SANS),
         (" to the trained store.",
          11.5, False, False, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Four ",
          11, False, False, BODY, SANS),
         ("presentation bands",
          11, True, False, RED, SANS),
         (", anchored on the calibrated ",
          11, False, False, BODY, SANS),
         ("τ",
          11, False, True, DARK, MONO),
         ("  (not a hand-set cut):",
          11, False, False, BODY, SANS)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("grounded",        10.5, True,  False, GREEN, SANS),
         ("            d ≤ τ",
          10.5, False, False, DARK, MONO)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("embellishment",   10.5, True,  False, DARK, SANS),
         ("      mild drift",
          10.5, False, True, BODY, SANS)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("distortion",      10.5, True,  False, DARK, SANS),
         ("          meaningful error",
          10.5, False, True, BODY, SANS)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("fabrication",     10.5, True,  False, RED, SANS),
         ("        at the OOV ceiling (2.0)",
          10.5, False, True, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Adversarial probes: ",
          10.5, False, False, BODY, SANS),
         ("boundary",
          10.5, True, False, DARK, SANS),
         (" · ",
          10.5, False, False, BODY, SANS),
         ("near-miss",
          10.5, True, False, DARK, SANS),
         (" · ",
          10.5, False, False, BODY, SANS),
         ("elaboration",
          10.5, True, False, DARK, SANS),
         (".",
          10.5, False, False, BODY, SANS)],
    ],
    [[
        ("one failure code  ·  HALLUCINATION_DETECTED  ·  CRITICAL",
         10.5, True, True, RED, SANS),
    ]],
)

# arrow 1
arrow(slide, l1 + panel_w + Inches(0.05), t_top + Inches(1.67),
      Inches(0.35), Inches(0.40), BLUE)

# ---------- Panel 2: Calibration (Appendix G) ----------------------------
l2 = l1 + panel_w + gap
make_panel(
    l2, BLUE, BLUE_BG,
    "Calibration",
    "is confidence honest?",
    [
        [("A promise: ",
          11.5, False, False, BODY, SANS),
         ("90% confident",
          11.5, True, False, DARK, SANS),
         (" should be right ",
          11.5, False, False, BODY, SANS),
         ("90% of the time",
          11.5, True, False, DARK, SANS),
         (".",
          11.5, False, False, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Angular encoding on the unit circle:",
          11, True, False, BLUE, SANS)],
        [("     ",       10.5, False, False, DARK, MONO),
         ("θ_c = c · π",  11,   True, False, DARK, MONO),
         ("      confidence → angle",
          10.5, False, True, BODY, SANS)],
        [("     ",       10.5, False, False, DARK, MONO),
         ("θ_a = 0 or π", 11,   True, False, DARK, MONO),
         ("   wrong / right",
          10.5, False, True, BODY, SANS)],
        [("     ",       10.5, False, False, DARK, MONO),
         ("gap = |θ_c − θ_a| / π",
                         11,   True, False, DARK, MONO)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Symmetric penalty — ",
          10.5, False, False, BODY, SANS),
         ("overconfident",
          10.5, True, False, RED, SANS),
         (" and ",
          10.5, False, False, BODY, SANS),
         ("underconfident",
          10.5, True, False, DARK, SANS),
         (" both score high.",
          10.5, False, False, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Release-gate metrics:  ",
          11, True, False, DARK, SANS),
         ("ECE",
          11, True, False, BLUE, MONO),
         (" (mean gap)  ·  ",
          10.5, False, False, BODY, SANS),
         ("MCE",
          11, True, False, BLUE, MONO),
         (" (worst bin)",
          10.5, False, False, BODY, SANS)],
    ],
    [[
        ("reliability diagram  +  overconfidence ratio",
         10.5, True, True, BLUE, SANS),
    ]],
)

# arrow 2
arrow(slide, l2 + panel_w + Inches(0.05), t_top + Inches(1.67),
      Inches(0.35), Inches(0.40), GREEN)

# ---------- Panel 3: Multi-turn memory (Ch 17) ---------------------------
l3 = l2 + panel_w + gap
make_panel(
    l3, GREEN, GREEN_BG,
    "Multi-turn memory",
    "does the agent remember?",
    [
        [("Single-turn tests cannot see ",
          11, False, False, BODY, SANS),
         ("forgetting, drift, leakage",
          11, True, True, DARK, SANS),
         (".",
          11, False, False, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("The ",
          11, False, False, BODY, SANS),
         ("teach–delay–recall",
          11, True, False, GREEN, SANS),
         (" probe:",
          11, False, False, BODY, SANS)],
        [("  ",           10.5, False, False, DARK, SANS),
         ("teach",        10.5, True,  False, GREEN, MONO),
         ("           plant a fact into the session",
          10.5, False, True, BODY, SANS)],
        [("  ",           10.5, False, False, DARK, SANS),
         ("delay",        10.5, True,  False, GREEN, MONO),
         ("            ",
          10.5, False, False, BODY, SANS),
         ("n",            10.5, False, True, DARK, MONO),
         (" distractor turns in between",
          10.5, False, True, BODY, SANS)],
        [("  ",           10.5, False, False, DARK, SANS),
         ("recall",       10.5, True,  False, GREEN, MONO),
         ("         ask for it back, score the recall",
          10.5, False, True, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("MemoryRecallOracle scores on ",
          10.5, False, False, BODY, SANS),
         ("two signals",
          10.5, True, False, DARK, SANS),
         (":",
          10.5, False, False, BODY, SANS)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("recall geodesic",  10.5, True, False, DARK, SANS),
         ("   vs. expected triple",
          10.5, False, True, BODY, SANS)],
        [("  ·  ",           10.5, False, False, DARK, SANS),
         ("session tension",  10.5, True, False, DARK, SANS),
         ("    cross-turn consistency",
          10.5, False, True, BODY, SANS)],
        [(" ", 4, False, False, DARK, SANS)],
        [("Degradation flag fires on ",
          10.5, False, False, BODY, SANS),
         (">20% tension rise",
          10.5, True, False, GREEN, SANS),
         (" across the run.",
          10.5, False, False, BODY, SANS)],
    ],
    [[
        ("interaction scripts + per-turn assertions on top",
         10.5, True, True, GREEN, SANS),
    ]],
)

# -- Concrete results strip -----------------------------------------------
dd_top = Inches(5.70)
box(slide, Inches(0.5), dd_top, Inches(12.4), Inches(0.55), WHITE,
    line=FAINT, rounded=True)
txt(slide, Inches(0.7), dd_top + Inches(0.07), Inches(2.8), Inches(0.40), [[
    ("Concrete numbers:", 11, True, False, NAVY, SANS),
]], anchor=MSO_ANCHOR.MIDDLE)
txt(slide, Inches(3.3), dd_top + Inches(0.07), Inches(9.5), Inches(0.40), [[
    ("Ch 16  τ = 1.385, ceiling 2.0   ·   ",
     10.5, False, False, DARK, MONO),
    ("App G  ECE 0.34, MCE 0.54, overconf 78%",
     10.5, False, False, DARK, MONO),
    ("   ·   ",
     10.5, False, False, DARK, MONO),
    ("Ch 17  recall 8/20 at delay = 5",
     10.5, False, False, DARK, MONO),
    ("     ",
     10.5, False, False, DARK, MONO),
    ("(all thresholds calibrated, not hand-set)",
     10, False, True, MID, MONO),
]], anchor=MSO_ANCHOR.MIDDLE)

# -- Bottom punchline bar -------------------------------------------------
bar_top = Inches(6.42)
box(slide, Inches(0.5), bar_top, Inches(12.4), Inches(0.95), NAVY, rounded=True)
txt(slide, Inches(0.75), bar_top + Inches(0.10), Inches(11.9), Inches(0.35), [[
    ("Three probes, one manifold, one calibration file.  ",
     12, True, False, WHITE, SANS),
    ("Each protocol reuses the same geometry that verified the facts.",
     12, False, False, WHITE, SANS),
]])
txt(slide, Inches(0.75), bar_top + Inches(0.48), Inches(11.9), Inches(0.42), [[
    ("Hallucination bands are just distances.  Calibration gaps are just angles.  Memory recall is just geodesic plus tension.  ",
     11, False, True, WHITE, SANS),
    ("The specialised behaviour is specialised.  The measurement is not.",
     11, False, True, WHITE, SANS),
]])

out = "/home/asudjianto/jupyterlab/kg-memory/behavioral_slide.pptx"
prs.save(out)
print(f"Wrote: {out}")
