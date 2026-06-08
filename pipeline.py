import io
import re
from typing import List

from typing import Any as UploadFile

from image_utils import (
    load_image_from_bytes,
    pdf_bytes_to_images,
    process_single_image,
)


class UnsupportedFileTypeError(Exception):
    pass


class CorruptFileError(Exception):
    pass


async def process_files_to_pdf(files: List[UploadFile]) -> bytes:
    """Process a list of UploadFile objects and return a single PDF (bytes).

    The original upload order is preserved. PDFs are expanded page-by-page.
    """
    processed_pages = []  # list of PIL.Image (mode 'L')

    for upload in files:
        filename = upload.filename or ""
        content_type = (upload.content_type or "").lower()

        data = await upload.read()
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
                pil_page = process_single_image(img)
                processed_pages.append(pil_page)

        elif is_image:
            try:
                img = load_image_from_bytes(data)
            except Exception:
                raise CorruptFileError()

            pil_page = process_single_image(img)
            processed_pages.append(pil_page)
        else:
            raise UnsupportedFileTypeError()

    if not processed_pages:
        raise CorruptFileError()

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
