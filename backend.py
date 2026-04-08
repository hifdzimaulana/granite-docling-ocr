#!/usr/bin/env python3
"""Local VLM OCR server using IBM Granite Docling with MLX."""

import json
import os
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import fitz
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc import DocTagsDocument
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from transformers.image_utils import load_image

DEFAULT_PROMPT = "Convert this page to docling."

model = None
processor = None
config = None


def init_model():
    global model, processor, config
    print("Loading MLX model (this may take a few minutes on first run)...")
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
    print("Model loaded successfully!")


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text("text").strip()
        texts.append(text)
    doc.close()
    return texts


def process_image(
    image_path: str, context_text: str = None, custom_prompt: str = None
) -> str:
    pil_image = load_image(image_path)

    base_prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

    if context_text:
        prompt = f"""Raw text extracted from document:
{context_text}

Using the above text as accurate reference, {base_prompt}
Preserve accurate text from the raw extraction above while enriching with visual structure."""
    else:
        prompt = base_prompt

    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    output = ""
    for token in stream_generate(
        model, processor, formatted_prompt, [pil_image], max_tokens=4096, verbose=False
    ):
        output += token.text
        if "</doctag>" in token.text:
            break

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output], [pil_image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    return doc.export_to_markdown()


def process_pdf(pdf_path: str, custom_prompt: str = None) -> str:
    raw_texts = extract_text_from_pdf(pdf_path)
    results = []

    for i, text in enumerate(raw_texts):
        print(f"Processing page {i + 1}/{len(raw_texts)}...")
        img = pdf_page_to_image(pdf_path, i)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            markdown = process_image(
                tmp_path, context_text=text, custom_prompt=custom_prompt
            )
            results.append(markdown)
        finally:
            os.unlink(tmp_path)

    return "\n\n---\n\n".join(results)


class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/ocr":
            self.send_error(404, "Not found")
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        boundary = (
            content_type.split("boundary=")[1] if "boundary=" in content_type else None
        )

        if not boundary:
            self.send_error(400, "No boundary found")
            return

        body = self.rfile.read(content_length)
        parts = body.split(f"--{boundary}".encode())

        file_data = None
        filename = None
        custom_prompt = None
        for part in parts:
            if b"Content-Disposition" in part:
                if b'name="prompt"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        prompt_data = part[header_end + 4 :].strip()
                        if prompt_data and not prompt_data.startswith(b"------"):
                            custom_prompt = prompt_data.decode(
                                "utf-8", errors="ignore"
                            ).strip()
                elif b"filename=" in part:
                    name_start = part.find(b'filename="')
                    if name_start != -1:
                        name_start += 10
                        name_end = part.find(b'"', name_start)
                        filename = part[name_start:name_end].decode()
                    if (
                        b"Content-Type: image/" in part
                        or b"Content-Type: application/pdf" in part
                    ):
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            file_data = part[header_end + 4 :]
                            break

        if not file_data:
            self.send_error(400, "No file found")
            return

        is_pdf = filename and filename.lower().endswith(".pdf")

        try:
            if is_pdf:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                try:
                    markdown = process_pdf(tmp_path, custom_prompt=custom_prompt)
                    response = {
                        "success": True,
                        "markdown": markdown,
                        "type": "pdf",
                        "pages": len(extract_text_from_pdf(tmp_path)),
                    }
                finally:
                    os.unlink(tmp_path)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                try:
                    markdown = process_image(tmp_path, custom_prompt=custom_prompt)
                    response = {"success": True, "markdown": markdown, "type": "image"}
                finally:
                    os.unlink(tmp_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            import traceback

            traceback.print_exc()
            response = {"success": False, "error": str(e)}
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("static/index.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-Type", "text/css")
            self.end_headers()
            with open("static/style.css", "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    init_model()
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), OCRHandler)
    print(f"Server running at http://localhost:{port}")
    print("Open your browser to use the OCR interface")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
