"""
Text Extraction Service
=======================
Responsible for converting raw input (bytes, URL, file path) into plain text
that can be passed to the LLM parser.

Strategy by file type
---------------------
PDF   → pymupdf4llm  (handles both embedded-text and scanned pages via built-in
                       hybrid OCR; returns Markdown for better structure preservation)
Image → base64-encode the image and pass directly to the vision LLM
URL   → fetch the page HTML / file, then choose PDF or image path

The functions here are intentionally simple — each does one thing and can be
replaced independently without touching the rest of the pipeline.
"""
from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional
from pdf2image import convert_from_bytes, convert_from_path

import httpx
import pymupdf4llm
from app.models.paddle_connect import _get_ocr
from app.common.utils import _preprocess

logger = logging.getLogger(__name__)


# ── PDF extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF given as raw bytes.

    Uses pymupdf4llm which:
    - Returns embedded text as Markdown (tables, headers preserved)
    - Automatically runs OCR on scanned / image-only pages
    - Handles multi-column layouts

    Returns the full document as a single Markdown string.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        md_text: str = pymupdf4llm.to_markdown(tmp_path)
        # if not md_text.strip():
        #     pass
        return md_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_text_from_pdf_path(pdf_path: str) -> str:
    """Extract text from a PDF file on disk."""
    return pymupdf4llm.to_markdown(pdf_path)


# ── Image extraction (returns base64 for vision LLM) ───────────────────────

def image_bytes_to_base64(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """
    Encode image bytes as a data-URI string ready for multimodal LLM calls.
    e.g.  "data:image/jpeg;base64,/9j/4AAQ..."
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def load_image_bytes(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Return a dict that the LLM service can use as an image content block.
    Keeps the raw bytes + media type separate from the data-URI so the caller
    decides which format the LLM library expects.
    """
    return {
        "bytes": image_bytes,
        "base64_uri": image_bytes_to_base64(image_bytes, media_type),
        "media_type": media_type,
    }

def extract_text_from_image_bytes(image_bytes: bytes, media_type: str = "image/jpeg"):
    processed_img, _ = _preprocess(image_bytes)
    ocr = _get_ocr()
    results = ocr.predict(processed_img)

    full_text = ""

    for page in results:
        full_text += "\n".join(page["rec_texts"])

    return full_text

# ── URL fetching ────────────────────────────────────────────────────────────

def fetch_bytes_from_url(url: str, timeout: float = 30.0) -> tuple[bytes, str]:
    """
    Download a resource from a URL.

    Returns:
        (content_bytes, content_type)   e.g.  (b"...", "application/pdf")
    """
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "application/octet-stream")
    # Strip parameters: "application/pdf; charset=utf-8" → "application/pdf"
    content_type = content_type.split(";")[0].strip()
    return response.content, content_type


def extract_text_from_url(url: str) -> tuple[str, str]:
    """
    Fetch a URL and extract text from it.

    Returns:
        (extracted_text_or_base64_uri, input_type)
        input_type is one of: "pdf_text", "image_base64", "html_text"
    """
    content_bytes, content_type = fetch_bytes_from_url(url)

    if "pdf" in content_type:
        text = extract_text_from_pdf_bytes(content_bytes)
        return text, "pdf_text"

    if content_type.startswith("image/"):
        image_data = load_image_bytes(content_bytes, content_type)
        return image_data["base64_uri"], "image_base64"

    # Fallback: treat as HTML / plain text
    text = content_bytes.decode("utf-8", errors="replace")
    return text, "html_text"


# ── Unified entry point ─────────────────────────────────────────────────────

class ExtractionResult:
    """Container returned by extract_from_input."""

    def __init__(
        self,
        text: Optional[str] = None,
        image_base64_uri: Optional[str] = None,
        input_kind: str = "text",  # "text" | "image"
        page_count: Optional[int] = None,
    ):
        self.text = text
        self.image_base64_uri = image_base64_uri
        self.input_kind = input_kind
        self.page_count = page_count

    @property
    def is_image(self) -> bool:
        return self.input_kind == "image"

def extract_invoice_from_input(file_bytes, media_type):
    """
    Extract invoice details from PDF/Image.
    Image byte + media type ->
    1. PDF -> extract text to markdown
    2. Image -> convert to pdf then extract text to markdown
    Return the extracted object
    """
    if file_bytes is None:
        raise ValueError("Provide file_bytes")
    # ---- Bytes input ----
    if "pdf" in media_type:
        text = extract_text_from_pdf_bytes(file_bytes)
        return ExtractionResult(text=text, input_kind="text")

    if media_type.startswith("image/"):
        img_text = extract_text_from_image_bytes(file_bytes, media_type)
        return ExtractionResult(
            text=img_text, input_kind="image"
        )


def extract_from_input(
    *,
    file_bytes: Optional[bytes] = None,
    file_url: Optional[str] = None,
    media_type: str = "application/octet-stream",
) -> ExtractionResult:
    """
    Unified extraction function.  Accepts either raw bytes OR a URL.

    media_type hints the file type when bytes are provided.
    Typical values: "application/pdf", "image/jpeg", "image/png"

    The result always has either `.text` (for PDF/HTML) or
    `.image_base64_uri` (for images) populated.
    """
    if file_bytes is None and file_url is None:
        raise ValueError("Provide either file_bytes or file_url")

    # ---- URL input ----
    if file_url:
        raw_text, kind = extract_text_from_url(file_url)
        if kind == "image_base64":
            return ExtractionResult(image_base64_uri=raw_text, input_kind="image")
        return ExtractionResult(text=raw_text, input_kind="text")

    # ---- Bytes input ----
    if "pdf" in media_type:
        text = extract_text_from_pdf_bytes(file_bytes)
        return ExtractionResult(text=text, input_kind="text")

    if media_type.startswith("image/"):
        img_data = load_image_bytes(file_bytes, media_type)
        return ExtractionResult(
            image_base64_uri=img_data["base64_uri"], input_kind="image"
        )

    # Unknown binary – try decoding as text
    logger.warning("Unknown media_type %s, trying UTF-8 decode", media_type)
    text = file_bytes.decode("utf-8", errors="replace")
    return ExtractionResult(text=text, input_kind="text")
