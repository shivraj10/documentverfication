import logging

from extractors.base_extractor import BaseExtractor, load_prompts, normalize_gender, normalize_mobile
from models import DocumentData

logger = logging.getLogger(__name__)
_PROMPTS = load_prompts()

                                                 
class DocumentExtractor(BaseExtractor):
    """
    Extracts structured identity information from any Indian identity document.

    Supported documents (non-exhaustive):
      PAN Card, Driving Licence, Passport, Voter ID, Aadhaar Card
    """

    PROMPT: str = _PROMPTS["document"]["extraction"]

    def _confidence(self, data: dict) -> float:
        """
        Weighted confidence: name + id_number are critical (60%),
        dob / gender / mobile_number are optional (40%).
        """
        critical = ["name", "id_number"]
        optional = ["dob", "gender", "mobile_number"]

        # Calculate the score
        score = (
            sum(bool(data.get(f)) for f in critical) / len(critical) * 0.6
            + sum(bool(data.get(f)) for f in optional) / len(optional) * 0.4
        )
        return round(score, 2)

    async def extract(self, image_bytes: bytes) -> DocumentData:
        logger.info("Starting document extraction.")

        try:
            # Load the image
            image = self._load_image(image_bytes)
        except ValueError as exc:
            return DocumentData(errors=[str(exc)])

        try:
            # Use gemini to extract the text
            raw_text = await self._call_gemini(self.PROMPT, image)
        except Exception as exc:
            logger.error("Gemini API error: %s: %s", type(exc).__name__, exc)
            return DocumentData(errors=[f"Gemini API error: {exc}"])

        try:
            # Parse gemini response
            parsed = self._parse_response(raw_text)
        except ValueError as exc:
            return DocumentData(raw_text=raw_text, errors=[str(exc)])

        # Calculate the confidence score
        confidence = self._confidence(parsed)

        doc_data = DocumentData(
            document_type=parsed.get("document_type"),
            name=parsed.get("name"),
            dob=parsed.get("dob"),
            gender=normalize_gender(parsed.get("gender")),
            mobile_number=normalize_mobile(parsed.get("mobile_number")),
            id_number=parsed.get("id_number"),
            raw_text=raw_text,
            extraction_confidence=confidence,
        )

        logger.info(
            "Document extraction complete. Type: %s. Confidence: %.2f.",
            doc_data.document_type,
            confidence,
        )
        return doc_data