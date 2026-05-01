"""1-page slide: Six Failure Modes of Agentic AI.

Combines:
  - Substack essay "Six Ways Agentic AI Fails" (the six modes, examples)
  - Monograph Ch 1, Lab 1.3 + Table "Failure modes and their GMS primitives"
    (adds the detection primitive per mode).

Layout: 2-column x 3-row grid of failure-mode cards.
  Left column (red):   Reasoning failures - FM-1, FM-4, FM-5
  Right column (blue): Execution failures - FM-2, FM-3, FM-6
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

RED    = RGBColor(0xB2, 0x3A, 0x2F)   # reasoning failures
BLUE   = RGBColor(0x1F, 0x6F, 0xB5)   # execution failures

RED_BG   = RGBColor(0xFB, 0xE9, 0xE7)
BLUE_BG  = RGBColor(0xE6, 0xF0, 0xFA)

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


def fm_card(slide, left, top, width, height, accent, accent_bg,
            code, name, what, example, detect):
    """Draw a single failure-mode card."""
    # Body
    box(slide, left, top, width, height, fill=accent_bg, line=accent, rounded=True)
    # Header stripe
    box(slide, left, top, width, Inches(0.42), fill=accent, rounded=True)
    box(slide, left, top + Inches(0.30), width, Inches(0.12), fill=accent)
    txt(slide, left + Inches(0.15), top + Inches(0.02),
        width - Inches(0.3), Inches(0.38), [[
            (code + "  ",  11.5, True, False, WHITE, MONO),
            (name,         13,   True, False, WHITE, SANS),
        ]], anchor=MSO_ANCHOR.MIDDLE)

    # Body text
    body_top = top + Inches(0.50)
    body_h   = height - Inches(0.55)
    txt(slide, left + Inches(0.22), body_top,
        width - Inches(0.44), body_h, [
            [(what, 10.5, False, False, DARK, SANS)],
            [(" ", 3, False, False, DARK, SANS)],
            [("e.g.  ",    10, True, True, MID, SANS),
             (example, 10, False, True, MID, SANS)],
            [(" ", 3, False, False, DARK, SANS)],
            [("Detect:  ", 10, True, False, accent, SANS),
             (detect,      10, False, False, DARK, MONO)],
        ])


# --- Slide ---------------------------------------------------------------
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Navy stripe
box(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.10), NAVY)

# Title
txt(slide, Inches(0.5), Inches(0.22), Inches(12.4), Inches(0.55), [[
    ("Six Failure Modes of Agentic AI", 30, True, False, NAVY, SANS),
    ("   —   where traditional validation falls silent",
     16, False, False, MID, SANS),
]])
txt(slide, Inches(0.5), Inches(0.78), Inches(12.4), Inches(0.28), [[
    ("Agus Sudjianto  &  Aldo Lau", 11, False, True, MID, SANS),
]])

# One-line framing
txt(slide, Inches(0.5), Inches(1.12), Inches(12.4), Inches(0.42), [[
    ("Traditional validation asks if the ",
     13, False, False, BODY, SANS),
    ("answer",
     13.5, True, True, DARK, SANS),
    (" is right.  Agentic validation asks if the agent ",
     13, False, False, BODY, SANS),
    ("behaved correctly across its full operational scope",
     13.5, True, True, NAVY, SANS),
    (".",
     13, False, False, BODY, SANS),
]])

# Column labels
col_w   = Inches(6.0)
gap_x   = Inches(0.35)
left_l  = Inches(0.55)
right_l = left_l + col_w + gap_x

txt(slide, left_l, Inches(1.64), col_w, Inches(0.32), [[
    ("REASONING FAILURES",  11, True, False, RED, SANS),
    ("     what the agent concludes",
     10, False, True, MID, SANS),
]])
txt(slide, right_l, Inches(1.64), col_w, Inches(0.32), [[
    ("EXECUTION FAILURES",  11, True, False, BLUE, SANS),
    ("     what the agent does",
     10, False, True, MID, SANS),
]])

# ----- Cards -------------------------------------------------------------
card_h   = Inches(1.42)
row_gap  = Inches(0.12)
row_top  = [Inches(1.95),
            Inches(1.95) + card_h + row_gap,
            Inches(1.95) + 2 * (card_h + row_gap)]

# Left column (REASONING, red)
fm_card(slide, left_l, row_top[0], col_w, card_h, RED, RED_BG,
        "FM-1", "Factual Hallucination",
        "Asserts facts absent from the source — or contradicts it.",
        "Returns training AUC 0.8934 when asked for test AUC 0.8412.",
        "score_triple   →   geodesic d > τ")

fm_card(slide, left_l, row_top[1], col_w, card_h, RED, RED_BG,
        "FM-4", "Inconsistent Multi-Hop Reasoning",
        "Fails to chain across tables, documents, or reasoning steps.",
        "Returns raw row for MDL-009 instead of tracing the dependency chain.",
        "check_holonomy   →   path vs. direct > ε")

fm_card(slide, left_l, row_top[2], col_w, card_h, RED, RED_BG,
        "FM-5", "Confidence Miscalibration",
        "Confidence disconnected from actual accuracy — wrong answers bypass scrutiny.",
        "Reports 8/10 confidence uniformly, regardless of correctness.",
        "Exact storage  →  ECE, MCE, reliability diagram")

# Right column (EXECUTION, blue)
fm_card(slide, right_l, row_top[0], col_w, card_h, BLUE, BLUE_BG,
        "FM-2", "Tool Misuse",
        "Wrong tool, wrong arguments, wrong order.",
        "Calls data_lookup + risk_check on an escalate request — never triggers escalate.",
        "query_triples   →   wrong relation in trace")

fm_card(slide, right_l, row_top[1], col_w, card_h, BLUE, BLUE_BG,
        "FM-3", "Policy Violation",
        "Skips required approval or performs a forbidden action.  Infrastructure-healthy, governance-broken.",
        "Automatic escalation without supervisor review.",
        "query_triples   →   forbidden action predicate")

fm_card(slide, right_l, row_top[2], col_w, card_h, BLUE, BLUE_BG,
        "FM-6", "State Corruption Across Turns",
        "Forgets, confuses, or contradicts earlier turns in a session.",
        "Fact taught at turn 1 → \"no relevant memory found\" at turn 5.",
        "lookup_enm + tension   →   teach-delay-recall")

# -- Bottom punchline bar -------------------------------------------------
bar_top = Inches(6.42)
box(slide, Inches(0.5), bar_top, Inches(12.4), Inches(0.95), NAVY, rounded=True)
txt(slide, Inches(0.75), bar_top + Inches(0.10), Inches(11.9), Inches(0.35), [[
    ("Each mode needs a ",
     12, False, False, WHITE, SANS),
    ("purpose-built probe",
     12, True, False, WHITE, SANS),
    (".  Each probe reads the ",
     12, False, False, WHITE, SANS),
    ("same calibrated manifold",
     12, True, False, WHITE, SANS),
    (".  Each verdict is auditable back to a source sentence.",
     12, False, False, WHITE, SANS),
]])
txt(slide, Inches(0.75), bar_top + Inches(0.48), Inches(11.9), Inches(0.42), [[
    ("Agentic validation is not an afterthought to model evaluation.  ",
     11, False, True, WHITE, SANS),
    ("It is a structured, multi-dimensional discipline with one tool per failure mode — six modes, six tools, one geometry.",
     11, False, True, WHITE, SANS),
]])

out = "/home/asudjianto/jupyterlab/kg-memory/failure_modes_slide.pptx"
prs.save(out)
print(f"Wrote: {out}")
