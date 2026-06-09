# Agent Integration Guide: OCR Image Processing Module

This document is designed for AI agents, language models, or developers looking to integrate the `ocr` processing module into an external codebase, application, or different backend (e.g., Django, Flask, Celery, Lambda).

## 🚀 Architecture Overview

This module is **completely decoupled from FastAPI**. It performs CPU-bound heavy OCR, perspective correction, image scaling, and PDF conversion natively.

The core pipeline entry point is in `pipeline.py`. It accepts generic Python `bytes` wrapped in a simple `dataclass` and outputs `bytes` representing the final PDF.

---

## 📦 Core Interface

### 1. `DocumentData` (Input Structure)
Any external backend must gather its file bytes and pass them into the pipeline using the `DocumentData` dataclass found in `pipeline.py`:

```python
from dataclasses import dataclass

@dataclass
class DocumentData:
    data: bytes              # The raw bytes of the image or PDF
    filename: str = ""       # Used for error tracing
    content_type: str = ""   # e.g., 'image/png', 'application/pdf'
```

### 2. `process_documents_to_pdf()` (The Engine)
```python
async def process_documents_to_pdf(documents: List[DocumentData], check_blur: bool = True) -> bytes:
```

- **`documents`**: A list of `DocumentData` objects. If you pass a multi-page PDF, it will be automatically split into individual page images.
- **`check_blur`**: If set to `True`, the pipeline uses Tesseract OCR to ensure the document has readable text. If no text is found, it raises `BlurryImageError`. To process visual-only data (like drawings), pass `check_blur=False`.
- **Returns**: A single `bytes` object containing the fully processed, lossless-compressed, stitched PDF.

---

## 🛠️ Integration Example (Any Python Backend)

Here is a standard example of how to invoke the pipeline from ANY python environment (e.g., a background worker, script, or alternative web framework):

```python
import asyncio
from pipeline import process_documents_to_pdf, DocumentData, BlurryImageError

async def handle_invoices(raw_file_bytes: bytes, filename: str, mime_type: str):
    # 1. Package the incoming data into the standard module format
    doc = DocumentData(
        data=raw_file_bytes,
        filename=filename,
        content_type=mime_type
    )
    
    try:
        # 2. Run the asynchronous pipeline
        # (Uses asyncio.to_thread internally to avoid blocking the event loop)
        pdf_bytes = await process_documents_to_pdf([doc], check_blur=True)
        
        # 3. Do whatever you want with the resulting PDF bytes
        with open("output.pdf", "wb") as f:
            f.write(pdf_bytes)
            
        return "Success!"
        
    except BlurryImageError as e:
        print(f"Validation Failed: {e}")
    except Exception as e:
        print(f"Processing Error: {e}")

# To run it in a script:
# asyncio.run(handle_invoices(b'...', "invoice.jpg", "image/jpeg"))
```

## ⚠️ Important Details for Agents

1. **Memory Warnings**: The pipeline chunks parallel processing in batches of `15` to prevent RAM explosions. If deploying to a memory-constrained environment (like AWS Lambda 512MB), reduce the `batch_size` inside `pipeline.py`.
2. **Decompression Bombs**: The `PIL.Image.MAX_IMAGE_PIXELS` limit has been disabled in `image_utils.py` to allow the processing of massive 100MP+ scanned documents.
3. **Lossless Compression**: The output PDF preserves the maximum visual fidelity because the module temporarily caches the frames as `PNG` with `optimize=True` before using `img2pdf` to wrap them losslessly.
