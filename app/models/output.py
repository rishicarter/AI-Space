"""
Pydantic models that define the structured JSON output for the API.

DocumentType controls which parser is used:
  - INVOICE  → InvoiceOutput  (invoice fields + DB lookups)
  - GENERAL  → GeneralOutput  (free-form content summary)
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INVOICE = "invoice"
    GENERAL = "general"


# ── Invoice sub-models ──────────────────────────────────────────────────────

class LineItem(BaseModel):
    description: str = Field(..., description="Item description as on the invoice")
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    item_code: Optional[str] = Field(
        None, description="Internal item code looked up from the database or retrieved from database"
    )


class InvoiceOutput(BaseModel):
    document_type: DocumentType = DocumentType.INVOICE
    invoice_number: Optional[str] = Field(None, description="Invoice / bill number")
    invoice_date: Optional[str] = Field(None, description="Invoice date (ISO-8601 preferred)")
    vendor_name: Optional[str] = Field(None, description="Vendor / supplier name")
    vendor_code: Optional[str] = Field(
        None, description="Internal vendor code looked up from the database"
    )
    total_amount: Optional[float] = Field(None, description="Grand total on the invoice")
    currency: Optional[str] = Field(None, description="Currency code, e.g. USD, INR")
    line_items: list[LineItem] = Field(default_factory=list)
    raw_text_preview: Optional[str] = Field(
        None, description="First 500 chars of extracted text for debugging"
    )


# ── General document model ──────────────────────────────────────────────────

class GeneralOutput(BaseModel):
    document_type: DocumentType = DocumentType.GENERAL
    summary: str = Field(..., description="Brief summary of the document content")
    content: str = Field(..., description="Full extracted text or structured content")
    page_count: Optional[int] = None
    raw_text_preview: Optional[str] = Field(
        None, description="First 500 chars of extracted text for debugging"
    )


# ── API response wrapper ────────────────────────────────────────────────────

class OCRResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: InvoiceOutput | GeneralOutput
