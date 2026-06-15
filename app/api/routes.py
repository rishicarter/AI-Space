"""
OCR Pipeline API Routes
=======================
Provides two endpoints:

  POST /ocr/upload
    Accept a multipart file upload (PDF, JPG, PNG).
    Optional query param: document_type=invoice|general
    If omitted, the LLM auto-detects the document type.

  POST /ocr/url
    Accept a JSON body with a URL pointing to a PDF or image.
    Optional field: document_type

Both endpoints:
  1. Extract text / image data from the input
  2. Call the LLM parser synchronously
  3. Return a structured JSON response
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, HttpUrl

from app.models.output import DocumentType, OCRResponse
from app.services.extractor import extract_from_input, extract_invoice_from_input
# from app.services.extractor_v2 import extract_invoice_from_input
from app.services.parser import parse_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Pipeline"])


# ── Helpers ──────────────────────────────────────────────────────────────────

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
}


def _media_type_from_filename(filename: str) -> str:
    """Guess MIME type from file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        "pdf": "application/pdf",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }
    return mapping.get(ext, "application/octet-stream")


# ── Upload endpoint ──────────────────────────────────────────────────────────

@router.post(
    "/read_invoice",
    response_model=OCRResponse,
    summary="Read Invoice file and generate structured line items",
    description=(
        "Upload a PDF or image file with Invoice information in it."
    )
)
def read_invoice(
    file: UploadFile = File(..., description="PDF, JPG, or PNG file"),
):
    media_type = file.content_type or _media_type_from_filename(file.filename or "")

    if media_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {media_type}. Supported: {sorted(SUPPORTED_MIME_TYPES)}",
        )

    try:
        file_bytes = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")
    
    try:
        extractacted_text = extract_invoice_from_input(file_bytes, media_type)
    except Exception as ex:
        raise HTTPException(status_code=422, detail=f"Text extraction from Invoice failed.")
    try:
        result = parse_document(
            text=extractacted_text.text,
            # image_base64_uri=extractacted_text.image_base64_uri,
            document_type="invoice",
        )
    except Exception as exc:
        logger.exception("LLM parsing failed")
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {exc}")
    return OCRResponse(data=result)

@router.post(
    "/upload",
    response_model=OCRResponse,
    summary="Process an uploaded file (PDF or image)",
    description=(
        "Upload a PDF or image file. "
        "The API will extract text, classify the document, and return structured JSON. "
        "Pass `document_type=invoice` to skip auto-detection."
    ),
)
def ocr_upload(
    file: UploadFile = File(..., description="PDF, JPG, or PNG file"),
    document_type: Optional[DocumentType] = Query(
        None, description="Force document type. Omit to auto-detect."
    ),
):
    # Determine media type from content-type header or filename
    media_type = file.content_type or _media_type_from_filename(file.filename or "")

    if media_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {media_type}. Supported: {sorted(SUPPORTED_MIME_TYPES)}",
        )

    try:
        file_bytes = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    try:
        extraction = extract_from_input(file_bytes=file_bytes, media_type=media_type)
    except Exception as exc:
        logger.exception("Extraction failed for %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    try:
        result = parse_document(
            text=extraction.text,
            image_base64_uri=extraction.image_base64_uri,
            document_type=document_type,
        )
    except Exception as exc:
        logger.exception("LLM parsing failed")
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {exc}")

    return OCRResponse(data=result)


# ── URL endpoint ─────────────────────────────────────────────────────────────

class URLRequest(BaseModel):
    url: HttpUrl
    document_type: Optional[DocumentType] = None


@router.post(
    "/url",
    response_model=OCRResponse,
    summary="Process a document from a URL",
    description=(
        "Provide a URL pointing to a PDF or image. "
        "The API will download it, extract text, and return structured JSON."
    ),
)
def ocr_url(body: URLRequest):
    url_str = str(body.url)

    try:
        extraction = extract_from_input(file_url=url_str)
    except Exception as exc:
        logger.exception("URL extraction failed: %s", url_str)
        raise HTTPException(status_code=422, detail=f"Failed to fetch/extract from URL: {exc}")

    try:
        result = parse_document(
            text=extraction.text,
            image_base64_uri=extraction.image_base64_uri,
            document_type=body.document_type,
        )
    except Exception as exc:
        logger.exception("LLM parsing failed")
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {exc}")

    return OCRResponse(data=result)


# ── Health check ─────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
