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

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    print("ERROR: MODEL_PATH environment variable not set")
    print("Set it with: export MODEL_PATH=/path/to/model")
    sys.exit(1)

model = None
processor = None
config = None
active_jobs = {}
cancel_flags = {}


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

    if pix.width == 0 or pix.height == 0 or len(pix.samples) == 0:
        doc.close()
        raise ValueError(
            f"Page {page_num + 1} has no renderable content (empty or blank page)"
        )

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


def process_image_stream(
    job_id: str, image_path: str, context_text: str = None, custom_prompt: str = None
):
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
    chunk = ""
    for token in stream_generate(
        model, processor, formatted_prompt, [pil_image], max_tokens=4096, verbose=False
    ):
        if cancel_flags.get(job_id):
            return None

        output += token.text
        chunk += token.text

        if len(chunk) >= 50:
            yield chunk
            chunk = ""

        if "</doctag>" in token.text:
            break

    if chunk:
        yield chunk

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output], [pil_image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    return doc.export_to_markdown()


def process_image(
    image_path: str, context_text: str = None, custom_prompt: str = None
) -> str:
    result = None
    for chunk in process_image_stream(None, image_path, context_text, custom_prompt):
        if chunk is None:
            return ""
        result = chunk
    return result or ""


def process_pdf_stream(job_id: str, pdf_path: str, custom_prompt: str = None):
    raw_texts = extract_text_from_pdf(pdf_path)
    total_pages = len(raw_texts)
    results = []

    for i, text in enumerate(raw_texts):
        if cancel_flags.get(job_id):
            yield {"type": "cancelled", "data": "".join(results)}
            return

        yield {"type": "progress", "data": f"Processing page {i + 1}/{total_pages}..."}

        try:
            img = pdf_page_to_image(pdf_path, i)
        except ValueError as e:
            print(f"Skipping page {i + 1}: {e}")
            results.append(f"<!-- Page {i + 1}: Empty/blank page -->")
            yield {
                "type": "page_done",
                "data": f"<!-- Page {i + 1}: Empty/blank page -->",
            }
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            page_result = []
            for chunk in process_image_stream(
                job_id, tmp_path, context_text=text, custom_prompt=custom_prompt
            ):
                if chunk is None:
                    yield {"type": "cancelled", "data": "".join(results)}
                    return
                page_result.append(chunk)

            markdown = "".join(page_result)
            results.append(markdown)
            yield {"type": "page_done", "data": markdown}
        finally:
            os.unlink(tmp_path)

    final_result = "\n\n---\n\n".join(results)
    yield {"type": "done", "data": final_result}


def process_pdf(pdf_path: str, custom_prompt: str = None) -> str:
    result = ""
    for event in process_pdf_stream(None, pdf_path, custom_prompt):
        if event["type"] == "done":
            result = event["data"]
    return result


class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/ocr/stream":
            self._handle_stream()
            return
        elif self.path == "/api/ocr/cancel":
            self._handle_cancel()
            return
        elif self.path == "/api/ocr":
            pass
        else:
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

    def _handle_stream(self):
        import uuid

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

        job_id = str(uuid.uuid4())
        cancel_flags[job_id] = False
        is_pdf = filename and filename.lower().endswith(".pdf")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Job-ID", job_id)
        self.end_headers()

        try:
            if is_pdf:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                try:
                    for event in process_pdf_stream(
                        job_id, tmp_path, custom_prompt=custom_prompt
                    ):
                        if cancel_flags.get(job_id):
                            self.wfile.write(
                                f"event: cancelled\ndata: {json.dumps({'markdown': event.get('data', '')})}\n\n".encode()
                            )
                            self.wfile.flush()
                            break
                        if event["type"] == "progress":
                            self.wfile.write(
                                f"event: progress\ndata: {json.dumps({'message': event['data']})}\n\n".encode()
                            )
                            self.wfile.flush()
                        elif event["type"] in ["page_done", "done", "cancelled"]:
                            self.wfile.write(
                                f"event: {event['type']}\ndata: {json.dumps({'markdown': event['data']})}\n\n".encode()
                            )
                            self.wfile.flush()
                finally:
                    os.unlink(tmp_path)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                try:
                    self.wfile.write(
                        f"event: progress\ndata: {json.dumps({'message': 'Processing image...'})}\n\n".encode()
                    )
                    self.wfile.flush()

                    accumulated = ""
                    for chunk in process_image_stream(
                        job_id, tmp_path, custom_prompt=custom_prompt
                    ):
                        if chunk is None:
                            self.wfile.write(
                                f"event: cancelled\ndata: {json.dumps({'markdown': accumulated})}\n\n".encode()
                            )
                            self.wfile.flush()
                            break
                        accumulated += chunk
                        self.wfile.write(
                            f"event: chunk\ndata: {json.dumps({'markdown': chunk})}\n\n".encode()
                        )
                        self.wfile.flush()

                    if not cancel_flags.get(job_id):
                        self.wfile.write(
                            f"event: done\ndata: {json.dumps({'markdown': accumulated})}\n\n".encode()
                        )
                        self.wfile.flush()
                finally:
                    os.unlink(tmp_path)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.wfile.write(
                f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n".encode()
            )
            self.wfile.flush()
        finally:
            del cancel_flags[job_id]

    def _handle_cancel(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode())
            job_id = data.get("jobId")
            if job_id and job_id in cancel_flags:
                cancel_flags[job_id] = True
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"success": False, "error": "Job not found"}).encode()
                )
        except:
            self.send_error(400, "Invalid request")

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
