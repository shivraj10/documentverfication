import logging
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger(__name__)

@dataclass
class Settings:
    """Central configuration object loaded from environment variables."""

    GEMINI_API_KEY: str
    DEBUG: bool
    VALIDITY_THRESHOLD: float

    def __post_init__(self):
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
            )
        logger.info(
            f"Settings loaded — DEBUG={self.DEBUG}, "
            f"VALIDITY_THRESHOLD={self.VALIDITY_THRESHOLD}"
        )

@dataclass
class FieldResult:
    """Result of comparing a single field between Aadhaar and document."""

    field_name: str
    aadhaar_value: Optional[str]
    document_value: Optional[str]
    match: bool
    similarity: float
    note: Optional[str] = None


@dataclass
class VerificationReport:
    """Full verification report comparing Aadhaar with another document."""

    is_valid: bool
    overall_score: float 
    fields: list[FieldResult] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    mismatched_fields: list[str] = field(default_factory=list)
    skipped_fields: list[str] = field(default_factory=list)  # Fields missing in one/both
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "matched_fields": self.matched_fields,
            "mismatched_fields": self.mismatched_fields,
            "skipped_fields": self.skipped_fields,
            "field_details": [
                {
                    "field": r.field_name,
                    "aadhaar_value": r.aadhaar_value,
                    "document_value": r.document_value,
                    "match": r.match,
                    "similarity": r.similarity,
                    "note": r.note,
                }
                for r in self.fields
            ],
        }


@dataclass
class AadhaarData:
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    mobile_number: Optional[str] = None
    aadhaar_number: Optional[str] = None
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dob": self.dob,
            "gender": self.gender,
            "mobile_number": self.mobile_number,
            "aadhaar_number": self.aadhaar_number,
            "extraction_confidence": self.extraction_confidence,
            "errors": self.errors,
        }

@dataclass
class DocumentData:
    """Structured identity fields extracted from any supported document."""

    document_type: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    mobile_number: Optional[str] = None
    id_number: Optional[str] = None 
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "name": self.name,
            "dob": self.dob,
            "gender": self.gender,
            "mobile_number": self.mobile_number,
            "id_number": self.id_number,
            "extraction_confidence": self.extraction_confidence,
            "errors": self.errors,
        }
