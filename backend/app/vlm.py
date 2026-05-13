"""Vision-language model: extract structured text from an image.

Replaces the older Tesseract-based OCR for the /query/image endpoint. Tesseract
reads geographically (top-to-bottom, left-to-right across the entire canvas),
which jumbles structured layouts — e.g. a 2x2 grid of multiple-choice options
becomes a single interleaved blob of all four options. A vision-language
model can recognize visual sections (boxes, columns, labeled panels) and
preserve their logical order, the way ChatGPT or Claude would.

We call a local Ollama vision model (default: ``llama3.2-vision:11b``) via
its chat API with the image embedded as a base64-encoded attachment. The
prompt instructs the model to act as a faithful, layout-preserving
transcriber — not as an analyst.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx
from loguru import logger

MAX_BYTES = 10 * 1024 * 1024  # 10 MB upper bound per attachment

EXTRACT_SYSTEM_PROMPT = """\
You are a precise text extractor. Read all visible text from the image and
reproduce it as faithfully as possible while preserving the visual structure.

RULES:
1. If the image has distinct visual sections (boxes, panels, columns,
   labeled options), label each clearly using the visible labels when they
   exist (e.g. "A)", "B)", "Question 5:", "Step 1:").
2. Keep text within a section together. Never interleave text from different
   sections, even if they sit side by side on the page.
3. Within each section, follow the natural reading order (top to bottom,
   left to right inside that section's bounds).
4. Preserve emphasis when it carries meaning — wrap visibly bold or large
   words in **double asterisks**.
5. Output ONLY the transcribed text. No preamble, no commentary, no
   description of the image, no "Here is the transcription".
"""


class VlmError(ValueError):
    """Raised when an image upload can't be processed by the vision model."""


@dataclass(frozen=True)
class VlmResult:
    text: str
    model: str


async def extract_text_from_image(
    image_bytes: bytes,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2-vision:11b",
    timeout: float = 120.0,
) -> VlmResult:
    """Send the image to a local Ollama vision model and return its transcription."""
    if not image_bytes:
        raise VlmError("Empty image upload.")
    if len(image_bytes) > MAX_BYTES:
        raise VlmError(f"Image too large ({len(image_bytes)} > {MAX_BYTES} bytes).")

    b64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Transcribe every piece of text in this image, preserving its visual structure.",
                "images": [b64],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(f"{base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise VlmError(f"Could not reach the vision model at {base_url}: {exc}") from exc

    if response.status_code != 200:
        body = response.text[:500]
        raise VlmError(
            f"Vision model returned {response.status_code}: {body}"
        )

    data = response.json()
    text = (data.get("message") or {}).get("content", "").strip()
    logger.info(
        f"VLM ({model}) extracted {len(text)} chars from a {len(image_bytes)}-byte image"
    )
    return VlmResult(text=text, model=model)
