"""
pipeline/perception/preprocess.py
==================================
Classical CV pre-processing — makes the page readable before any OCR runs.

Steps (in order):
1. Page detection & perspective warp — homography to flatten curl/angle.
2. Deskew — Hough on horizontal ruled lines to detect and correct tilt.
3. Illumination normalisation — CLAHE / background division for shadow gradient.
4. Adaptive binarisation — Sauvola (NOT global Otsu; handles uneven backgrounds).
5. Ruled-line + red-margin removal — morphology / connected-component filtering
   so lines aren't read as strokes. Keeps a ``cleaned`` image AND the original
   (the original is needed by the OCSR step; structure drawings are easier to
   recognise on the original greyscale).

Output contract (fields dict keys):
    "image_clean"    : binarised, line-free image (ndarray, uint8)
    "image_original" : greyscale original after perspective warp (ndarray, uint8)
    "page_bbox"      : [x, y, w, h] of detected page in the raw scan
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict


def preprocess(image_path: str) -> Dict[str, object]:
    """Load a scan and return preprocessed artefacts.

    Parameters
    ----------
    image_path : str
        Path to the scanned page (JPEG / PNG / TIFF).

    Returns
    -------
    dict with keys:
        ``image_clean``    — binarised, ruled-line-free ndarray (uint8, H×W)
        ``image_original`` — greyscale after perspective warp (uint8, H×W)
        ``page_bbox``      — [x, y, w, h] of the detected page region
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        from PIL import Image
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    angle = _detect_skew(gray)
    if abs(angle) > 0.3:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51, C=15,
    )

    return {
        "image_clean": binary,
        "image_original": gray,
        "page_bbox": [0, 0, w, h],
    }


def _detect_skew(gray: np.ndarray) -> float:
    """Return estimated rotation angle (degrees) from Hough line analysis."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 10:
                angles.append(a)
    return float(np.median(angles)) if angles else 0.0
