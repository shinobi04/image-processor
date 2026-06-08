# Invoice Preprocessing API - Environment Setup

This project uses Conda for environment management. The service requires system-level Tesseract OCR to be installed (used for orientation detection).

Conda + pip setup (recommended)

```bash
# Create and activate conda env
conda create -n invoice-api python=3.10 -y
conda activate invoice-api

# Install Tesseract via conda-forge (system binary)
conda install -c conda-forge tesseract -y

# Install Python dependencies (pip is used to get latest compatible packages)
pip install -r requirements.txt

# Verify tesseract is available
tesseract --version
```

Notes
- If you prefer installing Tesseract via your OS package manager, that is fine (e.g. `brew install tesseract` on macOS or `sudo apt install tesseract-ocr` on Debian/Ubuntu).
- `pillow-heif` integrates HEIC/HEIF support into Pillow so Pillow can open HEIC files.
- `pymupdf` (fitz) is used to rasterize PDF pages for processing.
- `img2pdf` is used to assemble processed images into a single PDF.
