import io
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3-flash-preview"
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


def load_prompts(path: Path = _PROMPTS_PATH) -> dict:
    """Load prompts from a YAML file. Returns the full prompts dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#  Standalone normalizers (reusable without instantiating a class) 

def normalize_gender(gender: Optional[str]) -> Optional[str]:
    """Normalize raw gender string to Male / Female / Transgender."""
    if not gender:
        return None
    g = gender.strip().lower()
    if g in ("male", "m"):
        return "Male"
    if g in ("female", "f"):
        return "Female"
    if g in ("transgender", "trans", "other"):
        return "Transgender"
    return gender.strip().title()


def normalize_mobile(mobile: Optional[str]) -> Optional[str]:
    """Strip country code and validate 10-digit Indian mobile number."""
    if not mobile:
        return None
    digits = re.sub(r"\D", "", str(mobile))
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    logger.warning("Invalid mobile number discarded: '%s'", mobile)
    return None

#  Abstract base extractor
class BaseExtractor(ABC):
    """
    Abstract base for Gemini Vision extractors.

    Subclasses must implement:
      - PROMPT  (class-level str)
      - extract(image_bytes) -> dataclass instance
    """

    MODEL_NAME: str = GEMINI_MODEL

    def __init__(self, api_key: str) -> None:
        logger.info("Initializing %s with Gemini Vision.", self.__class__.__name__)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.MODEL_NAME)
        logger.info("Gemini model loaded: %s", self.MODEL_NAME)

    #  Shared helpers

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Convert raw bytes to a PIL Image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            logger.debug("Image loaded: size=%s, mode=%s", image.size, image.mode)
            return image
        except Exception as exc:
            logger.error("Failed to load image: %s", exc)
            raise ValueError(f"Invalid image data: {exc}") from exc

    async def _call_gemini(self, prompt: str, image: Image.Image) -> str:
        """Send prompt + image to Gemini and return raw response text."""
        logger.info("Sending image to Gemini Vision API.")
        response = await self.model.generate_content_async(
            [prompt, image],
            generation_config={"temperature": 0.1},
        )
        logger.info("Gemini response received. Feedback: %s", response.prompt_feedback)
        raw = response.text
        logger.debug("Raw response (first 300 chars): %s", raw[:300])
        return raw

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini response, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
        try:
            parsed = json.loads(text)
            logger.debug("Parsed response: %s", parsed)
            return parsed
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc

    @abstractmethod
    async def extract(self, image_bytes: bytes):
        """Extract structured data from a document image."""