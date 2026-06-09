from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from typing import List
import logging

from pipeline import process_documents_to_pdf, DocumentData, UnsupportedFileTypeError, CorruptFileError, BlurryImageError

app = FastAPI(title="Invoice Preprocessing API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/process-invoices")
async def process_invoices(
    files: List[UploadFile] = File(...),
    check_blur: bool = Query(False, description="Enable or disable blur and blank page detection")
):
    """Accept multiple files (images and PDFs), process them, and return a single PDF."""
    if not files:
        raise HTTPException(status_code=400, detail={"error": "No files uploaded"})

    try:
        documents = []
        for f in files:
            data = await f.read()
            documents.append(DocumentData(data=data, filename=f.filename, content_type=f.content_type))
        pdf_bytes = await process_documents_to_pdf(documents, check_blur=check_blur)
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
