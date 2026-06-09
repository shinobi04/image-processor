import io
import asyncio
import re
from typing import List

import pytesseract

from dataclasses import dataclass

from image_utils import (
    load_image_from_bytes,
    pdf_bytes_to_images,
    process_single_image,
)


class UnsupportedFileTypeError(Exception):
    pass


class CorruptFileError(Exception):
    pass


class BlurryImageError(Exception):
    pass


def _process_and_ocr(img, filename, check_blur=True):
    """Helper function to run CPU-bound processing and OCR in a separate thread."""
    pil_page = process_single_image(img)
    
    if check_blur:
        MIN_CHAR_COUNT = 10  # Adjust this to change "blur/blank page" sensitivity
        text = pytesseract.image_to_string(pil_page).strip()
        if not text:
            raise BlurryImageError(f"Image or page from {filename} is blurry or contains no readable text.")
    return pil_page


@dataclass
class DocumentData:
    data: bytes
    filename: str = ""
    content_type: str = ""


async def process_documents_to_pdf(documents: List[DocumentData], check_blur: bool = True) -> bytes:
    """Process a list of DocumentData objects and return a single PDF (bytes).

    The original order is preserved. PDFs are expanded page-by-page.
    """
    raw_images_to_process = []  # list of tuples: (np.ndarray, filename)

    for doc in documents:
        filename = doc.filename or ""
        content_type = (doc.content_type or "").lower()

        data = doc.data
        if not data:
            raise CorruptFileError()

        is_pdf = "pdf" in content_type or filename.lower().endswith(".pdf")

        # Image MIME types we explicitly support
        image_mime_indicators = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/tiff",
            "image/heic",
            "image/heif",
        ]

        is_image = any(m in content_type for m in image_mime_indicators) or bool(re.search(r"\.(jpe?g|png|webp|tiff?|heic|heif)$", filename, re.IGNORECASE))

        if is_pdf:
            try:
                page_images = pdf_bytes_to_images(data)
            except Exception:
                raise CorruptFileError()

            for img in page_images:
                raw_images_to_process.append((img, filename))

        elif is_image:
            try:
                img = load_image_from_bytes(data)
            except Exception:
                raise CorruptFileError()

            raw_images_to_process.append((img, filename))
        else:
            raise UnsupportedFileTypeError()

    if not raw_images_to_process:
        raise CorruptFileError()

    # Process images concurrently in batches of 15 to control memory usage
    processed_pages = []
    batch_size = 15
    for i in range(0, len(raw_images_to_process), batch_size):
        batch = raw_images_to_process[i:i + batch_size]
        tasks = [asyncio.to_thread(_process_and_ocr, img, fname, check_blur) for img, fname in batch]
        # Await the batch; gather preserves the original order
        results = await asyncio.gather(*tasks)
        processed_pages.extend(results)

    # Convert processed PIL images to PDF using img2pdf (expects binary file-like objects)
    import img2pdf

    buffers = []
    for pil in processed_pages:
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        buffers.append(buf)

    try:
        pdf_bytes = img2pdf.convert(buffers)
    except Exception:
        # fallback using PIL's PDF writer
        out = io.BytesIO()
        first, rest = processed_pages[0], processed_pages[1:]
        first.save(out, format="PDF", save_all=True, append_images=rest)
        pdf_bytes = out.getvalue()

    return pdf_bytes
