"""
invoice_extraction.py
─────────────────────
Production-ready extraction layer for invoice OCR pipelines.

Supports
  • Native PDFs   — text layer extracted directly via pymupdf4llm (no OCR cost)
  • Scanned PDFs  — rasterised page-by-page → Tesseract
  • Any image     — JPEG, PNG, TIFF, BMP, WebP (incl. animated + RGBA),
                    GIF, ICO, PPM … anything Pillow can decode
  • Mixed-script  — Latin / Devanagari / auto-routed with OSD pre-pass

All paths return an ExtractionResult with .text set.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
import pymupdf          # fitz
import pymupdf4llm
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# TESSERACT CONFIGS
# ──────────────────────────────────────────────────────────────────────────────

# English-only invoices — fastest, most accurate for single-script docs
_CFG_ENG = (
    "--oem 3 "
    "--psm 6 "
    "-l eng "
    "-c preserve_interword_spaces=1 "
    "-c textord_heavy_nr=1 "
    "-c edges_max_children_per_outline=40 "
    # "-c tessedit_char_whitelist="
    # "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # "0123456789 .,/-:()@#$%&*+=<>[]{}|\\\"'₹$€£% "
)

# Hindi-only Devanagari invoices
_CFG_HIN = (
    "--oem 3 "
    "--psm 6 "
    "-l hin "
    "-c preserve_interword_spaces=1 "
    "-c textord_heavy_nr=1 "
)

# Mixed-script (eng + hin) — psm 3 lets Tesseract segment blocks by script first
_CFG_MULTI = (
    "--oem 3 "
    "--psm 3 "
    "-l eng+hin "
    "-c preserve_interword_spaces=1 "
    "-c textord_heavy_nr=1 "
    # No char whitelist: Devanagari codepoints won't fit in ASCII whitelist
)

# pymupdf4llm.to_markdown() production kwargs — shared across all PDF calls
_PYMUPDF_KWARGS: dict = dict(
    header=False,             # Drop repetitive page headers (logo, doc title)
    footer=False,             # Drop page numbers / legal footers
    hdr_info=False,           # Disable font-size heading detection; invoices have no semantic headings
    ignore_code=True,         # Prevent tabular data being wrapped in ```code``` blocks
    ignore_images=True,       # Logos/stamps carry no extraction value
    ignore_graphics=False,    # Keep vector graphics — table borders are vectors
    graphics_limit=500,       # Safety cap against pathological vector counts
    table_strategy="lines_strict",  # Ruling-line table detection; best for printed invoices
                                    # Fall back to "lines" for noisy scans if needed
    fontsize_limit=5,         # Filter watermarks / invisible micro-text (default=3)
    use_ocr=True,             # Auto-OCR pages with no selectable text
    force_ocr=False,          # Never force-OCR native PDFs — degrades quality
    ocr_dpi=300,              # 300 DPI for OCR; default 150 misses fine print/GST numbers
    ocr_language="eng",       # Override per-call when non-English PDFs detected
    page_chunks=False,        # True only when per-page metadata is needed downstream
    show_progress=False,
)

# Minimum text characters to consider a page "native" (has a real text layer)
_NATIVE_TEXT_THRESHOLD = 50


# ──────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionResult:
    """Container returned by extract_invoice_from_input."""

    def __init__(
        self,
        text: Optional[str] = None,
        image_base64_uri: Optional[str] = None,
        input_kind: str = "text",   # "text" | "image"
        page_count: Optional[int] = None,
        ocr_token_estimate: Optional[int] = None,
        detected_script: Optional[str] = None,
    ):
        self.text = text
        self.image_base64_uri = image_base64_uri
        self.input_kind = input_kind
        self.page_count = page_count
        self.ocr_token_estimate = ocr_token_estimate
        self.detected_script = detected_script

    @property
    def is_image(self) -> bool:
        return self.input_kind == "image"

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(kind={self.input_kind!r}, "
            f"pages={self.page_count}, "
            f"~{self.ocr_token_estimate} tokens, "
            f"script={self.detected_script!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_bytes(file_bytes: bytes, media_type: str = "") -> Image.Image:
    """
    Decode any image format (including WebP, animated WebP, RGBA/P-mode)
    into a plain RGB PIL Image safe for Tesseract and cv2.

    Handles:
      • JPEG, PNG, TIFF, BMP, GIF, ICO, PPM, PGM, TGA
      • WebP — static and animated (takes frame 0)
      • RGBA / LA / P-mode — composited against white (prevents black bg)
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot decode image (media_type={media_type!r}): {exc}") from exc

    # Animated WebP / GIF — take the first frame only
    if getattr(img, "n_frames", 1) > 1:
        logger.debug("Animated image detected (%d frames) — using frame 0", img.n_frames)
        img.seek(0)
        img = img.copy()

    # Normalise mode to RGB
    if img.mode in ("RGBA", "LA"):
        # Composite over white to preserve contrast on transparent invoices
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == "P":
        # Palette mode — convert via RGBA to keep any transparency info
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline optimised for invoice images.

      1. Upscale to ≥ 2000px on longest side (Tesseract prefers ~300 DPI)
      2. Grayscale conversion
      3. Deskew via Hough-line detection
      4. Non-local means denoise
      5. Adaptive Gaussian thresholding (handles uneven scan lighting)
      6. Morphological close (fills broken strokes in thin fonts)
    """
    img = np.array(pil_img.convert("RGB"))

    # 1. Upscale low-res inputs
    h, w = img.shape[:2]
    if max(h, w) < 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Deskew
    gray = _deskew(gray)

    # 4. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 5. Adaptive threshold — handles shadows, yellowed paper, uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    # 6. Morphological close — fills minor stroke gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(binary)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct minor rotation (≤ 10°) using Hough-line detection."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is not None:
        angles = [line[0][1] for line in lines]
        skew_deg = (float(np.median(angles)) - np.pi / 2) * (180 / np.pi)
        if abs(skew_deg) < 10:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), skew_deg, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    return gray


# ──────────────────────────────────────────────────────────────────────────────
# SCRIPT DETECTION + OCR ROUTING
# ──────────────────────────────────────────────────────────────────────────────

def _detect_script(pil_img: Image.Image) -> tuple[str, int]:
    """
    Run Tesseract OSD pass to detect dominant script and orientation.
    Returns (script_name, rotation_degrees).

    NOTE: osd must NOT be passed as a language in -l flag.
    It is an internal Tesseract script-detector, not a language pack.
    Passing '-l eng+hin+osd' causes errors in most Tesseract builds.
    """
    try:
        osd = pytesseract.image_to_osd(
            pil_img, output_type=pytesseract.Output.DICT
        )
        return osd.get("script", "Latin"), osd.get("rotate", 0)
    except pytesseract.TesseractError as exc:
        logger.warning("OSD failed (%s) — defaulting to Latin/0°", exc)
        return "Latin", 0


def _ocr_image(pil_img: Image.Image, force_multilingual: bool = False) -> tuple[str, str]:
    """
    OSD-aware OCR routing:
      • Runs a fast OSD pre-pass to detect script + orientation
      • Corrects rotation if needed
      • Routes to the appropriate Tesseract config

    Returns (ocr_text, detected_script).
    """
    # Pre-process first so OSD/OCR sees a clean image
    clean = _preprocess_for_ocr(pil_img)

    script, rotation = _detect_script(clean)
    logger.debug("OSD result: script=%s, rotation=%d°", script, rotation)

    # Correct orientation
    if rotation != 0:
        clean = clean.rotate(-rotation, expand=True)

    if force_multilingual:
        config = _CFG_MULTI
        script = "mixed"
    elif script == "Devanagari":
        config = _CFG_HIN
    elif script == "Latin":
        config = _CFG_ENG
    else:
        # Unknown / mixed scripts — use multi-language config with psm 3
        config = _CFG_MULTI
        script = "mixed"

    text = pytesseract.image_to_string(clean, config=config)
    return text, script


# ──────────────────────────────────────────────────────────────────────────────
# PDF UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _is_native_pdf(doc: pymupdf.Document) -> bool:
    """
    Returns True if the PDF has a real selectable text layer.
    Checks the first (up to 3) pages to guard against mixed docs
    where page 1 is a cover image but the rest is native text.
    """
    pages_to_check = min(3, len(doc))
    for i in range(pages_to_check):
        if len(doc[i].get_text().strip()) >= _NATIVE_TEXT_THRESHOLD:
            return True
    return False


def _extract_native_pdf(doc: pymupdf.Document) -> str:
    """
    Extract text from a native (digitally-created) PDF using pymupdf4llm.
    This is ~1000× faster than OCR and produces cleaner output.

    Key params (see _PYMUPDF_KWARGS for full rationale):
      • header/footer=False     — strips repeated navigation noise
      • hdr_info=False          — no fake ## headings on invoice fields
      • ignore_code=True        — prevents line-item tables becoming ```code```
      • table_strategy=lines_strict — accurate ruling-line table detection
      • force_ocr=False         — NEVER force OCR on native PDFs; degrades output
    """
    return pymupdf4llm.to_markdown(doc, **_PYMUPDF_KWARGS)


def _extract_scanned_pdf(doc: pymupdf.Document) -> str:
    """
    Rasterise each page of a scanned PDF and OCR individually.
    Uses fitz pixmap at 300 DPI → PIL → preprocess → Tesseract.

    300 DPI is the Tesseract sweet spot; lower values miss fine print
    and small currency amounts common in invoices.
    """
    pages_text: list[str] = []

    for page_num, page in enumerate(doc):
        # Render at 300 DPI (scale factor = 300/72 ≈ 4.17)
        mat = pymupdf.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        text, script = _ocr_image(pil_img)
        logger.debug("Scanned PDF page %d: script=%s, ~%d chars", page_num + 1, script, len(text))

        if text.strip():
            pages_text.append(text.strip())

    return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)


def _extract_mixed_pdf(doc: pymupdf.Document) -> str:
    """
    Some PDFs mix native text pages with scanned pages (e.g. a cover scan
    followed by typed pages). Handle each page independently.
    """
    pages_text: list[str] = []

    for page_num, page in enumerate(doc):
        native_text = page.get_text().strip()

        if len(native_text) >= _NATIVE_TEXT_THRESHOLD:
            # Native page — use pymupdf4llm on just this page
            page_md = pymupdf4llm.to_markdown(doc, pages=[page_num], **_PYMUPDF_KWARGS)
            pages_text.append(page_md.strip())
        else:
            # Scanned page — rasterise + OCR
            mat = pymupdf.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text, _ = _ocr_image(pil_img)
            if text.strip():
                pages_text.append(text.strip())

    return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC EXTRACTION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf_bytes(file_bytes: bytes) -> tuple[str, int]:
    """
    Smart PDF extractor — auto-detects whether each page is native or scanned.

    Strategy:
      • All native  → pymupdf4llm (fast, zero OCR)
      • All scanned → page-by-page Tesseract at 300 DPI
      • Mixed       → per-page routing

    Returns (text, page_count).
    """
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)

    if _is_native_pdf(doc):
        logger.info("PDF type: native (%d pages)", page_count)
        text = _extract_native_pdf(doc)
    else:
        # Check if it's fully scanned or a mix
        native_pages = sum(
            1 for page in doc
            if len(page.get_text().strip()) >= _NATIVE_TEXT_THRESHOLD
        )
        if native_pages == 0:
            logger.info("PDF type: fully scanned (%d pages)", page_count)
            text = _extract_scanned_pdf(doc)
        else:
            logger.info(
                "PDF type: mixed (%d native / %d scanned)",
                native_pages, page_count - native_pages,
            )
            text = _extract_mixed_pdf(doc)

    doc.close()
    return text, page_count


def extract_text_from_image_bytes(
    file_bytes: bytes,
    media_type: str = "",
    force_multilingual: bool = False,
) -> tuple[str, str]:
    """
    Universal image OCR — handles any Pillow-decodable format.

    Supported formats include (non-exhaustive):
      JPEG, PNG, TIFF, BMP, WebP (static + animated), GIF,
      ICO, PPM, PGM, PBM, TGA, DDS, PCX, SGI, SPIDER

    Returns (ocr_text, detected_script).
    """
    pil_img = _load_image_bytes(file_bytes, media_type)
    return _ocr_image(pil_img, force_multilingual=force_multilingual)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def extract_invoice_from_input(
    file_bytes: bytes,
    media_type: str,
    force_multilingual: bool = False,
) -> ExtractionResult:
    """
    Universal invoice extractor.

    Parameters
    ----------
    file_bytes        : Raw bytes of the uploaded file.
    media_type        : MIME type string, e.g. "application/pdf", "image/webp".
    force_multilingual: Force eng+hin mixed-script OCR regardless of OSD result.
                        Useful when you know the invoice is bilingual.

    Returns
    -------
    ExtractionResult with:
      .text                — extracted / OCR'd text
      .input_kind          — "text" (PDF) | "image"
      .page_count          — number of pages/frames processed
      .ocr_token_estimate  — rough LLM token cost (~chars / 4)
      .detected_script     — "Latin" | "Devanagari" | "mixed" | None
    """
    if file_bytes is None:
        raise ValueError("file_bytes must not be None")
    if not media_type:
        raise ValueError("media_type must be provided (e.g. 'application/pdf')")

    # ── PDF path ──────────────────────────────────────────────────────────────
    if "pdf" in media_type:
        text, page_count = extract_text_from_pdf_bytes(file_bytes)

        # Token guard — log warning for unexpectedly large extractions
        token_estimate = len(text) // 4
        if token_estimate > 3000:
            logger.warning(
                "PDF extraction ~%d tokens across %d page(s) — "
                "consider splitting before LLM call",
                token_estimate, page_count,
            )

        return ExtractionResult(
            text=text,
            input_kind="text",
            page_count=page_count,
            ocr_token_estimate=token_estimate,
            detected_script=None,  # N/A for native PDF; set for scanned in future
        )

    # ── Image path ────────────────────────────────────────────────────────────
    if media_type.startswith("image/"):
        img_text, script = extract_text_from_image_bytes(
            file_bytes,
            media_type=media_type,
            force_multilingual=force_multilingual,
        )

        token_estimate = len(img_text) // 4
        if token_estimate > 3000:
            logger.warning(
                "Image OCR output ~%d tokens — consider splitting the image",
                token_estimate,
            )

        return ExtractionResult(
            text=img_text,
            input_kind="image",
            page_count=1,
            ocr_token_estimate=token_estimate,
            detected_script=script,
        )

    raise ValueError(
        f"Unsupported media_type {media_type!r}. "
        "Expected 'application/pdf' or 'image/*'."
    )