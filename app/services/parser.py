"""
LLM Parsing Service
===================
Takes extracted text (or an image base64 URI) and returns a structured
InvoiceOutput or GeneralOutput by calling the configured LLM synchronously.

Model configuration (from app/core/config.py):
  - LLM_PROVIDER = "openrouter"  → ChatOpenRouter  (dev / free tier)
  - LLM_PROVIDER = "anthropic"   → ChatAnthropic   (prod)

The pipeline is:
  prompt → LLM → raw JSON string → Pydantic model

We use LangChain's .with_structured_output() where the model supports it,
falling back to a manual JSON extraction for models that don't.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Union
import os

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.db.lookup import get_item_code, get_vendor_code
from app.models.output import DocumentType, GeneralOutput, InvoiceOutput, LineItem

logger = logging.getLogger(__name__)


# ── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm(vision: bool = False):
    """
    Return the configured LangChain chat model.

    vision=True  → use a multimodal model (needed for raw image inputs)
    vision=False → may use a cheaper text-only model
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),#settings.ANTHROPIC_API_KEY,
            max_tokens=2048,
        )

    # Default: openrouter
    from langchain_openrouter import ChatOpenRouter
    # For vision tasks always use the vision model; text tasks can use cheaper model
    model_name = (
        settings.OPENROUTER_MODEL if vision else settings.OPENROUTER_TEXT_MODEL
    )
    return ChatOpenRouter(
        model=model_name,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),#settings.OPENROUTER_API_KEY,
        max_tokens=2048, reasoning={"effort": "none"}, 
        openrouter_provider={"order": ["google-ai-studio", "open-inference/bf16"]}
    )


# ── Prompts ──────────────────────────────────────────────────────────────────

INVOICE_SYSTEM_PROMPT = """You are a precise invoice data extraction assistant.
Extract the following fields from the document and return ONLY valid JSON — no markdown fences, no explanation.

Required JSON structure:
{
  "invoice_number": "<string or null>",
  "invoice_date": "<ISO-8601 date string or null>",
  "vendor_name": "<string or null>",
  "total_amount": <number or null>,
  "currency": "<3-letter code or null>",
  "line_items": [
    {
      "description": "<string>",
      "quantity": <number or null>,
      "unit_price": <number or null>,
      "total_price": <number or null>,
      "item_code": <string or null>
    }
  ]
}

Rules:
- Dates must be ISO-8601 (YYYY-MM-DD) if possible
- Amounts must be numeric (no currency symbols)
- line_items must be an array even if empty
- item_code should be present below the item description in the document, put the complete item code in string
- Return null for missing fields, never omit keys
"""

GENERAL_SYSTEM_PROMPT = """You are a document analysis assistant.
Analyse the provided document content and return ONLY valid JSON — no markdown fences, no explanation.

Required JSON structure:
{
  "summary": "<1-3 sentence summary of the document>",
  "content": "<full extracted text or structured representation>"
}
"""

DOCUMENT_TYPE_SYSTEM_PROMPT = """Classify this document. Reply with ONLY one word: "invoice" or "general"."""


# ── Document type detection ──────────────────────────────────────────────────

def detect_document_type(text_preview: str) -> DocumentType:
    """
    Ask the LLM to classify the document as invoice or general.
    Falls back to 'general' on any error.
    """
    llm = _build_llm(vision=False)
    try:
        messages = [
            SystemMessage(content=DOCUMENT_TYPE_SYSTEM_PROMPT),
            HumanMessage(content=text_preview[:2000]),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip().lower()
        if "invoice" in answer:
            return DocumentType.INVOICE
    except Exception as exc:
        logger.warning("Document type detection failed: %s", exc)
    return DocumentType.GENERAL


# ── JSON extraction helper ───────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Pull the first valid JSON object out of a raw LLM response string.
    Handles responses wrapped in ```json ... ``` fences or extra prose.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No valid JSON found in LLM response: {raw[:300]}")


# ── Invoice parsing ──────────────────────────────────────────────────────────

def parse_invoice(
    text: str | None = None,
    image_base64_uri: str | None = None,
) -> InvoiceOutput:
    """
    Call the LLM to extract invoice fields, then enrich with DB lookups.

    Accepts either plain text or an image base64 URI (for scanned invoices
    passed as images).
    """
    vision = image_base64_uri is not None
    llm = _build_llm(vision=vision)

    # Build message content
    if vision:
        content = [
            {"type": "text", "text": "Extract invoice data from this image."},
            {"type": "image_url", "image_url": {"url": image_base64_uri}},
        ]
    else:
        content = f"Extract invoice data from the following document:\n\n{text}"

    messages = [
        SystemMessage(content=INVOICE_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    raw_json = _extract_json(response.content)

    # Build line items with DB-resolved item codes
    line_items: list[LineItem] = []
    for item_dict in raw_json.get("line_items", []):
        if item_dict.get("item_code", None) is None:
            temp_item_code = get_item_code(item_dict.get("description", "")) 
        else:
            temp_item_code = item_dict.get("item_code", "")
        item = LineItem(
            description=item_dict.get("description", ""),
            quantity=item_dict.get("quantity"),
            unit_price=item_dict.get("unit_price"),
            total_price=item_dict.get("total_price"),
            item_code=temp_item_code,
        )
        line_items.append(item)

    vendor_name: str | None = raw_json.get("vendor_name")
    vendor_code = get_vendor_code(vendor_name) if vendor_name else None

    preview = (text or "")[:100] if text else None

    return InvoiceOutput(
        invoice_number=raw_json.get("invoice_number"),
        invoice_date=raw_json.get("invoice_date"),
        vendor_name=vendor_name,
        vendor_code=vendor_code,
        total_amount=raw_json.get("total_amount"),
        currency=raw_json.get("currency"),
        line_items=line_items,
        raw_text_preview=preview,
    )


# ── General document parsing ─────────────────────────────────────────────────

def parse_general(
    text: str | None = None,
    image_base64_uri: str | None = None,
    page_count: int | None = None,
) -> GeneralOutput:
    """Call the LLM to summarise and return the document content."""
    vision = image_base64_uri is not None
    llm = _build_llm(vision=vision)

    if vision:
        content = [
            {"type": "text", "text": "Describe and extract all text from this image."},
            {"type": "image_url", "image_url": {"url": image_base64_uri}},
        ]
    else:
        content = f"Analyse and summarise the following document:\n\n{text}"

    messages = [
        SystemMessage(content=GENERAL_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    raw_json = _extract_json(response.content)

    preview = (text or "")[:500] if text else None

    return GeneralOutput(
        summary=raw_json.get("summary", ""),
        content=raw_json.get("content", text or ""),
        page_count=page_count,
        raw_text_preview=preview,
    )


# ── Unified parse entry point ────────────────────────────────────────────────

def parse_document(
    text: str | None = None,
    image_base64_uri: str | None = None,
    document_type: DocumentType | None = None,
    page_count: int | None = None,
) -> Union[InvoiceOutput, GeneralOutput]:
    """
    Main entry point for the LLM parsing step.

    If document_type is not explicitly provided, auto-detect it from the text.
    Then dispatch to the appropriate parser.
    """
    # Auto-detect type if not supplied
    if document_type is None:
        preview = (text or "")[:2000] if text else "Image document"
        document_type = detect_document_type(preview)
        logger.info("Auto-detected document type: %s", document_type)

    if document_type == DocumentType.INVOICE:
        return parse_invoice(text=text, image_base64_uri=image_base64_uri)
    else:
        return parse_general(
            text=text, image_base64_uri=image_base64_uri, page_count=page_count
        )
