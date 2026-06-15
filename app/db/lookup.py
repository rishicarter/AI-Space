"""
Database Lookup Service
=======================
Provides lookup functions for vendor codes and item codes.

These are STUB implementations — replace the bodies with real DB queries
(SQLAlchemy, SQLite, Postgres, etc.) without changing the signatures.

The rest of the pipeline calls only these two functions, so swapping the
implementation is a one-file change.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── In-memory stub data ─────────────────────────────────────────────────────
# Replace these dicts with actual DB calls.

_VENDOR_DB: dict[str, str] = {
    "Universal Networks (India) Pvt. Ltd.": "PTR-000357",
    "acme corp": "VND-001",
    "acme": "VND-001",
    "globex": "VND-002",
    "globex corporation": "VND-002",
    "initech": "VND-003",
    "umbrella": "VND-004",
    "umbrella corporation": "VND-004",
    "wayne enterprises": "VND-005",
    "Excel Telesonic India Pvt Ltd": "VC0003",
    "SuperStore": ""
}

_ITEM_DB: dict[str, str] = {
    "laptop": "ITM-1001",
    "notebook": "ITM-1001",
    "mouse": "ITM-1002",
    "keyboard": "ITM-1003",
    "monitor": "ITM-1004",
    "desk": "ITM-1005",
    "chair": "ITM-1006",
    "headset": "ITM-1007",
    "webcam": "ITM-1008",
    "usb hub": "ITM-1009",
    "printer": "ITM-1010",
    "paper": "ITM-1011",
    "pen": "ITM-1012",
    "stapler": "ITM-1013",
}


# ── Public lookup functions ─────────────────────────────────────────────────

def get_vendor_code(vendor_name: str) -> str | None:
    """
    Look up the internal vendor code for a given vendor name.

    Args:
        vendor_name: The vendor name extracted from the invoice.

    Returns:
        Vendor code string (e.g. "VND-001") or None if not found.

    TODO: Replace stub with:
        return db.query(Vendor).filter(
            Vendor.name.ilike(f"%{vendor_name}%")
        ).first()?.code
    """
    if not vendor_name:
        return None

    key = vendor_name.strip().lower()

    # Exact match first
    if key in _VENDOR_DB:
        return _VENDOR_DB[key]

    # Partial match (vendor name contains a known key or vice-versa)
    for db_key, code in _VENDOR_DB.items():
        if db_key in key or key in db_key:
            logger.debug("Vendor partial match: '%s' → '%s' (%s)", vendor_name, db_key, code)
            return code

    logger.info("Vendor not found in DB: '%s'", vendor_name)
    return None


def get_item_code(item_description: str) -> str | None:
    """
    Look up the internal item code for a given item description.

    Args:
        item_description: The line-item description from the invoice.

    Returns:
        Item code string (e.g. "ITM-1001") or None if not found.

    TODO: Replace stub with a real DB / product catalogue lookup.
    """
    if not item_description:
        return None

    key = item_description.strip().lower()

    # Exact match
    if key in _ITEM_DB:
        return _ITEM_DB[key]

    # Keyword match — any known item keyword in the description
    for db_key, code in _ITEM_DB.items():
        if db_key in key:
            logger.debug("Item partial match: '%s' → '%s' (%s)", item_description, db_key, code)
            return code

    logger.info("Item not found in DB: '%s'", item_description)
    return None
