# Granite Docling OCR

Local VLM OCR server using IBM Granite Docling with MLX acceleration for Apple Silicon.

## Features

- PDF and image OCR processing
- Uses IBM Granite Docling Vision-Language Model via MLX
- Web interface for uploading documents
- Extracts text and visual structure from documents

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3)
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Copy `.env.example` to `.env` and update `MODEL_PATH` to your model location:

```bash
cp .env.example .env
# Edit .env with your model path
```

Manage the server:

```bash
./server.sh start    # Start server
./server.sh status   # Check if running
./server.sh stop     # Stop server
./server.sh restart  # Restart server
```

Server runs at `http://localhost:8080`

## API

```
POST /api/ocr
- Accepts multipart/form-data with:
  - file: PDF or image file
  - prompt: optional custom prompt
- Returns JSON with markdown output
```