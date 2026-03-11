"""Helpers for canonical Shopify product URLs and deterministic hashing."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def canonicalize_product_url(
    product_url: str | None,
    *,
    myshopify_domain: str | None = None,
    product_handle: str | None = None,
) -> str | None:
    """Return a canonical product URL suitable for stable hashing."""

    url = (product_url or "").strip()
    if not url:
        domain = (myshopify_domain or "").strip()
        handle = (product_handle or "").strip().lstrip("/")
        if not domain or not handle:
            return None
        url = f"https://{domain}/products/{handle}"

    if "://" not in url:
        url = "https://" + url.lstrip("/")

    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def hash_product_url(url: str) -> str:
    """Return a SHA-256 hex digest for ``url``."""

    return hashlib.sha256(url.encode("utf-8")).hexdigest()
