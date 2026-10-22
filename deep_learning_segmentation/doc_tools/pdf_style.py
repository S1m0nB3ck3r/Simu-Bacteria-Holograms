# -*- coding: utf-8 -*-
"""
Style commun des PDF de deep_learning_segmentation/ (PIPELINE_UNET3D.pdf, HELP_TRAIN.pdf, HELP_INFERENCE.pdf) :
polices DejaVu (accents, µ, σ, ×), styles de paragraphe, tableaux, pied de page.
"""
import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
DL_DIR = os.path.dirname(HERE)
_fd = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
for name, f in (('DV', 'DejaVuSans.ttf'), ('DVB', 'DejaVuSans-Bold.ttf'), ('DVI', 'DejaVuSans-Oblique.ttf'),
                ('DVM', 'DejaVuSansMono.ttf')):
    pdfmetrics.registerFont(TTFont(name, os.path.join(_fd, f)))
registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')

INK = colors.HexColor('#1f2328')
MUTED = colors.HexColor('#57606a')
ACC = colors.HexColor('#2a78d6')
GRIDC = colors.HexColor('#d0d7de')
HEAD = colors.HexColor('#f6f8fa')
S = dict(
    title=ParagraphStyle('t', fontName='DVB', fontSize=20, leading=25, textColor=INK, spaceAfter=6),
    sub=ParagraphStyle('s', fontName='DV', fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=14),
    h1=ParagraphStyle('h1', fontName='DVB', fontSize=14, leading=18, textColor=INK, spaceBefore=6, spaceAfter=8,
                      keepWithNext=1),
    h2=ParagraphStyle('h2', fontName='DVB', fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4,
                      keepWithNext=1),
    body=ParagraphStyle('b', fontName='DV', fontSize=9.2, leading=13, textColor=INK, spaceAfter=5),
    small=ParagraphStyle('sm', fontName='DV', fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4),
    cap=ParagraphStyle('c', fontName='DVI', fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8),
    cell=ParagraphStyle('ce', fontName='DV', fontSize=7.6, leading=9.6, textColor=INK),
    cellb=ParagraphStyle('cb', fontName='DVB', fontSize=7.6, leading=9.6, textColor=INK),
    mono=ParagraphStyle('m', fontName='DVM', fontSize=7.4, leading=9.6, textColor=INK),
    bullet=ParagraphStyle('bu', fontName='DV', fontSize=9.2, leading=13, textColor=INK, leftIndent=12, bulletIndent=2,
                          spaceAfter=3),
)
W = A4[0] - 3.4 * cm


def esc(t):
    """Texte brut -> texte sûr pour Paragraph (balises < > &)."""
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def P(t, st='body'):
    return Paragraph(t, S[st])


def bullets(items):
    return [Paragraph(t, S['bullet'], bulletText='•') for t in items]


def img(path, width=W, max_h=None):
    from PIL import Image as PI
    w, h = PI.open(path).size
    hh = width * h / w
    if max_h and hh > max_h:
        width, hh = max_h * w / h, max_h
    return Image(path, width=width, height=hh)


def table(rows, widths, header=True, mono_cols=()):
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else
                     Paragraph(c, S['cellb'] if (header and i == 0) else (S['mono'] if (j in mono_cols and i > 0) else S['cell']))
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, GRIDC), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def footer_for(title):
    def footer(canv, doc):
        canv.saveState()
        canv.setFont('DV', 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(1.7 * cm, 1.1 * cm, title)
        canv.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, f"{doc.page}")
        canv.restoreState()
    return footer


def build(out, story, title, footer_text):
    from reportlab.platypus import SimpleDocTemplate
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title=title, author="Simon BECKER")
    f = footer_for(footer_text)
    doc.build(story, onFirstPage=f, onLaterPages=f)
    return out
