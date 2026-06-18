"""
pipeline — the extraction and validation pipeline.

Inter-stage contract
--------------------
Every value extracted from the page is wrapped in a ``Field`` envelope:

    {
        "value":      <str | float | bool | None>,
        "confidence": <float 0..1>,
        "provenance": <"ocr_text"|"ocr_math"|"ocsr"|"chem_resolve"|
                       "layout"|"normalize"|"classify"|"derived">,
        "bbox":       [x, y, width, height],   # source-image pixels
        # optional:
        "unit":       <str>,
        "raw":        <str>,  # literal text as written, before normalization
    }

``perception`` emits one ``fields: dict[str, Field]``; ``classify`` reads it;
``registry.dispatch`` routes to the matched validator; ``main`` assembles
schema-valid JSON.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


class Field(TypedDict, total=False):
    """The auditable envelope every extracted value carries across pipeline stages."""

    value: Any
    confidence: float
    provenance: str
    bbox: List[float]
    unit: str
    raw: str
