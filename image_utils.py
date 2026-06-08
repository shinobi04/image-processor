import io
import math
import re
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import fitz  # PyMuPDF
import pytesseract
import cv2


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load image bytes into an OpenCV BGR numpy array.

    Supports HEIC/HEIF via pillow-heif integration with Pillow.
    """
    buf = io.BytesIO(data)
    with Image.open(buf) as pil:
        # If multi-frame image (TIFF), select first frame
        try:
            pil.seek(0)
        except Exception:
            pass
        pil = pil.convert("RGB")
        arr = np.array(pil)
        # Return RGB numpy array (uint8)
        return arr


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """Render PDF bytes to a list of OpenCV BGR images (one per page)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        arr = np.array(img)
        images.append(arr)
    return images


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # compute height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Use OpenCV perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return interior angle at point b (in degrees) for triangle a-b-c."""
    # vectors
    v0 = a - b
    v1 = c - b
    # normalize
    nv0 = v0 / (np.linalg.norm(v0) + 1e-8)
    nv1 = v1 / (np.linalg.norm(v1) + 1e-8)
    cosang = np.clip(np.dot(nv0, nv1), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang


def _validate_quad_angles(pts: np.ndarray, min_angle: float = 45.0, max_angle: float = 135.0) -> bool:
    """Validate that a 4-point polygon has interior angles within [min_angle, max_angle].

    pts may be in any order; we order them first.
    """
    if pts is None or pts.shape != (4, 2):
        return False

    rect = _order_points(pts)
    # compute interior angles at each corner
    for i in range(4):
        prev = rect[(i - 1) % 4]
        curr = rect[i]
        nxt = rect[(i + 1) % 4]
        ang = _angle_between(prev, curr, nxt)
        if ang < min_angle or ang > max_angle:
            return False
    return True


def _get_tight_content_bbox(gray: np.ndarray, min_area: int = 1000, margin: int = 50) -> Optional[Tuple[int, int, int, int]]:
    """Return a tight bounding box (x,y,w,h) around textual/content regions in a grayscale image.

    Uses adaptive thresholding and morphological ops to cluster text into blocks, then
    returns the union bbox of sufficiently large contours. If nothing is found, returns None.
    """
    if gray is None or gray.size == 0:
        return None

    h, w = gray.shape[:2]

    # OpenCV-based approach: adaptive threshold + morphological grouping of text
    bs = 15 if max(h, w) > 200 else 11
    try:
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 10)
    except Exception:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morph to join text into blocks
    kx = max(3, int(w / 200))
    ky = max(3, int(h / 200))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    morph = cv2.dilate(thr, kernel, iterations=2)

    contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        boxes.append((x, y, x + ww, y + hh))

    if not boxes:
        return None

    # union boxes
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    # add margin and clip
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    ww = x2 - x1
    hh = y2 - y1
    # sanity: require the bbox to cover at least a small fraction of the image
    if ww * hh < 0.01 * w * h:
        return None

    return (x1, y1, ww, hh)


def _choose_rotation_by_ocr(pil_img: Image.Image, conf_threshold: int = 40) -> int:
    """Try 0/90/180/270 and return the rotation (degrees) that gives the most
    high-confidence OCR words. We rotate the image by -angle to simulate correcting it.
    """
    # Try four candidate orientations
    candidates = [0, 90, 180, 270]
    best = (0, -1, -1)  # (angle, count, total_conf)

    for ang in candidates:
        try:
            test = pil_img.rotate(-ang, expand=True)
            data = pytesseract.image_to_data(test, output_type=pytesseract.Output.DICT)
        except Exception:
            continue

        texts = data.get("text", [])
        confs = data.get("conf", [])
        cnt = 0
        tot = 0
        for t, c in zip(texts, confs):
            if not t or len(t) < 3 or not re.search(r"\w", t):
                continue
            try:
                ic = int(c)
            except Exception:
                continue
            if ic >= conf_threshold:
                cnt += 1
                tot += ic

        if cnt > best[1] or (cnt == best[1] and tot > best[2]):
            best = (ang, cnt, tot)

    return int(best[0])


def detect_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to find the largest 4-corner contour in the image.

    Returns 4x2 array of float points or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize for speed if very large
    h, w = gray.shape
    resizing = False
    scale = 1.0
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        resizing = True
    else:
        gray_small = gray

    blur = cv2.GaussianBlur(gray_small, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            if resizing:
                pts = pts / scale
            # area guard: ensure detected quad is large enough
            rect = _order_points(pts)
            poly_area = 0.5 * np.abs(np.dot(rect[:, 0], np.roll(rect[:, 1], 1)) - np.dot(rect[:, 1], np.roll(rect[:, 0], 1)))
            if poly_area >= 0.15 * (h * w):
                return pts

    # Fallback: try morphological large-contour -> minAreaRect
    try:
        small = gray_small if resizing else gray
        thr = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            box = cv2.boxPoints(cv2.minAreaRect(c))
            pts = np.array(box, dtype="float32")
            if resizing:
                pts = pts / scale
            rect = _order_points(pts)
            poly_area = 0.5 * np.abs(np.dot(rect[:, 0], np.roll(rect[:, 1], 1)) - np.dot(rect[:, 1], np.roll(rect[:, 0], 1)))
            if poly_area >= 0.15 * (h * w):
                return pts
    except Exception:
        pass

    return None


def _parse_osd_rotation(osd: str) -> int:
    """Parse tesseract OSD output and return (rotation_degrees, orientation_confidence).

    Example OSD:
      Rotate: 270
      Orientation confidence: 5.22
    """
    rot = 0
    conf = 0.0
    m = re.search(r"Rotate:\s*(\d+)", osd)
    if m:
        rot = int(m.group(1))
    mc = re.search(r"Orientation confidence:\s*([0-9.]+)", osd)
    if mc:
        try:
            conf = float(mc.group(1))
        except Exception:
            conf = 0.0
    return rot, conf


def process_single_image(image_bgr: np.ndarray) -> Image.Image:
    """Run the full preprocessing pipeline on a single BGR image and return a PIL grayscale Image.

    Steps: detect/crop document, perspective correction, grayscale, orientation, portrait enforcement,
    denoise, CLAHE contrast, and mild sharpening.
    """
    # Step 1: detect and perspective-correct
    pts = detect_document_corners(image_bgr)
    warped = image_bgr
    if pts is not None:
        # validate geometry to avoid extreme distortions
        try:
            if _validate_quad_angles(pts):
                # check polygon area to avoid tiny background quads
                rect = _order_points(pts)
                poly_area = 0.5 * np.abs(np.dot(rect[:, 0], np.roll(rect[:, 1], 1)) - np.dot(rect[:, 1], np.roll(rect[:, 0], 1)))
                img_area = image_bgr.shape[0] * image_bgr.shape[1]
                if poly_area >= 0.15 * img_area:
                    warped = _four_point_transform(image_bgr, pts)
                else:
                    # too small; fallback to safe bounding rect crop
                    pts_int = pts.astype(int)
                    x1 = int(np.min(pts_int[:, 0]))
                    y1 = int(np.min(pts_int[:, 1]))
                    x2 = int(np.max(pts_int[:, 0]))
                    y2 = int(np.max(pts_int[:, 1]))
                    hh, ww = image_bgr.shape[:2]
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(ww, x2 + 10)
                    y2 = min(hh, y2 + 10)
                    warped = image_bgr[y1:y2, x1:x2]
            else:
                # invalid angles: fallback to safe rect
                pts_int = pts.astype(int)
                x1 = int(np.min(pts_int[:, 0]))
                y1 = int(np.min(pts_int[:, 1]))
                x2 = int(np.max(pts_int[:, 0]))
                y2 = int(np.max(pts_int[:, 1]))
                hh, ww = image_bgr.shape[:2]
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(ww, x2 + 10)
                y2 = min(hh, y2 + 10)
                warped = image_bgr[y1:y2, x1:x2]
        except Exception:
            warped = image_bgr

    # Step 2: convert to grayscale (uint8)
    try:
        if warped.ndim == 3 and warped.shape[2] == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped.copy().astype(np.uint8)
    except Exception:
        # last resort
        gray = np.array(Image.fromarray(warped).convert("L"), dtype=np.uint8)

    # Step 3: orientation detection with Tesseract OSD
    pil_gray = Image.fromarray(gray)
    rotation = 0
    rot_conf = 0.0
    # First try Tesseract OSD
    try:
        osd = pytesseract.image_to_osd(pil_gray)
        rotation, rot_conf = _parse_osd_rotation(osd)
    except Exception:
        rotation, rot_conf = 0, 0.0

    # Apply OSD rotation only if confidence is reasonably high
    if rotation and rotation % 360 != 0 and rot_conf >= 30.0:
        try:
            pil_gray = pil_gray.rotate(-rotation, expand=True)
            gray = np.array(pil_gray)
        except Exception:
            pass
    else:
        # Either OSD suggested no rotation or confidence is low: use OCR-based selection
        try:
            ocr_choice = _choose_rotation_by_ocr(pil_gray)
            if ocr_choice and ocr_choice % 360 != 0:
                pil_gray = pil_gray.rotate(-ocr_choice, expand=True)
                gray = np.array(pil_gray)
        except Exception:
            pass


    # Step 5: denoising
    try:
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    except Exception:
        denoised = gray

    # Step 6: CLAHE contrast enhancement
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
    except Exception:
        contrast = denoised

    # Step 7: mild sharpening (unsharp mask via addWeighted)
    try:
        blur = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)
        final = np.clip(sharpened, 0, 255).astype("uint8")
    except Exception:
        final = contrast

    # Convert to PIL grayscale image
    # Before finalizing, attempt a tight content crop to remove excess margins
    try:
        arr_final = final
        bbox = _get_tight_content_bbox(arr_final)
        if bbox is not None:
            x, y, w, h = bbox
            arr_final = arr_final[y : y + h, x : x + w]
        pil_final = Image.fromarray(arr_final).convert("L")
    except Exception:
        pil_final = Image.fromarray(final).convert("L")
    return pil_final
