from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from typing import List
import logging

from pipeline import process_files_to_pdf, UnsupportedFileTypeError, CorruptFileError, BlurryImageError

app = FastAPI(title="Invoice Preprocessing API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/process-invoices")
async def process_invoices(files: List[UploadFile] = File(...)):
    """Accept multiple files (images and PDFs), process them, and return a single PDF."""
    if not files:
        raise HTTPException(status_code=400, detail={"error": "No files uploaded"})

    try:
        pdf_bytes = await process_files_to_pdf(files)
    except UnsupportedFileTypeError:
        raise HTTPException(status_code=415, detail={"error": "Unsupported file type"})
    except CorruptFileError:
        raise HTTPException(status_code=422, detail={"error": "Unable to process file"})
    except BlurryImageError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    except Exception as exc:  # unexpected
        logger.exception("Unexpected error while processing files")
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})

    return Response(content=pdf_bytes, media_type="application/pdf")
