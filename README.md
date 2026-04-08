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

Set the model path (download from [MLX Granite Docling](https://huggingface.co/ibm/granite-vision-llm-v2.0-3b-mlx)):

```bash
export MODEL_PATH=/path/to/granite-vision-llm-v2.0-3b-mlx
python backend.py
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