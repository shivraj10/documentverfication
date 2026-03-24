import logging

from extractors.base_extractor import BaseExtractor, load_prompts, normalize_gender, normalize_mobile
from models import AadhaarData
logger = logging.getLogger(__name__)

_PROMPTS = load_prompts()

 
class AadhaarExtractor(BaseExtractor):
    """Extracts structured fields from an Aadhaar card image."""

    PROMPT: str = _PROMPTS["aadhaar"]["extraction"]

    _FIELDS = ["name", "dob", "gender", "mobile_number", "aadhaar_number"]

    def _confidence(self, data: dict) -> float:
        return round(sum(bool(data.get(f)) for f in self._FIELDS) / len(self._FIELDS), 2)

    async def extract(self, image_bytes: bytes) -> AadhaarData:
        try:
            # Load Image
            image = self._load_image(image_bytes)
        except ValueError as exc:
            return AadhaarData(errors=[str(exc)])

        try:
            # Use Gemini to extract text
            raw_text = await self._call_gemini(self.PROMPT, image)
        except Exception as exc:
            return AadhaarData(errors=[f"Gemini API error: {exc}"])

        try:
            # Parse the response
            parsed = self._parse_response(raw_text)
        except ValueError as exc:
            return AadhaarData(raw_text=raw_text, errors=[str(exc)])

        # Sending the parsed data
        data = AadhaarData(
            name=parsed.get("name"),
            dob=parsed.get("dob"),
            gender=normalize_gender(parsed.get("gender")),
            mobile_number=normalize_mobile(parsed.get("mobile_number")),
            aadhaar_number=parsed.get("aadhaar_number"),
            raw_text=raw_text,
            extraction_confidence=self._confidence(parsed),
        )

        if data.aadhaar_number and len(data.aadhaar_number) != 12:
            data.errors.append("Invalid Aadhaar number")

        return data