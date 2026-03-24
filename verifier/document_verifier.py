import logging
import re

from typing import Optional
from difflib import SequenceMatcher

from extractors.aadhar_extractor import AadhaarData
from extractors.document_extractor import DocumentData
from models import FieldResult, VerificationReport

logger = logging.getLogger(__name__)


class DocumentVerifier:
    """
    Compares Aadhaar card data with data extracted from another document.

    Strategy:
    - Name:          Fuzzy match (handles minor OCR errors / initials)
    - DOB:           Normalized date comparison (exact)
    - Gender:        Exact match (normalized)
    - Mobile Number: Exact match (normalized 10-digit)

    A document is considered VALID if the overall score >= threshold (default 0.75)
    and name is not a hard mismatch (similarity < 0.5).
    """

    FIELD_WEIGHTS = {
        "name":          0.40,
        "dob":           0.35,
        "gender":        0.15,
        "mobile_number": 0.10,
    }

    VALIDITY_THRESHOLD = 0.75
    NAME_SIMILARITY_THRESHOLD = 0.75

    def __init__(self, validity_threshold: float = 0.75):
        self.validity_threshold = validity_threshold
        logger.info(
            f"DocumentVerifier initialized. Validity threshold: {self.validity_threshold}"
        )

    # --- Normalization helpers ---

    def _normalize_text(self, text: Optional[str]) -> Optional[str]:
        """Lowercase, strip extra spaces, remove punctuation for comparison."""
        if not text:
            return None
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def _normalize_date(self, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize date to DD/MM/YYYY.
        Handles: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, YYYY/MM/DD.
        """
        if not date_str:
            return None
        m = re.match(r"(\d{2})[/\-](\d{2})[/\-](\d{4})", date_str.strip())
        if m:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        m = re.match(r"(\d{4})[/\-](\d{2})[/\-](\d{2})", date_str.strip())
        if m:
            return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
        logger.warning(f"Could not normalize date: {date_str}")
        return date_str.strip()

    def _normalize_mobile(self, mobile: Optional[str]) -> Optional[str]:
        """Strip non-digits and country code, return clean 10-digit number."""
        if not mobile:
            return None
        digits = re.sub(r"\D", "", mobile)
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        return digits if len(digits) == 10 else None

    def _string_similarity(self, a: Optional[str], b: Optional[str]) -> float:
        """Compute similarity ratio between two strings (0.0 to 1.0)."""
        if not a or not b:
            return 0.0
        return round(SequenceMatcher(None, a, b).ratio(), 3)

    # --- Field comparators ---

    def _compare_name(
        self, aadhaar_name: Optional[str], doc_name: Optional[str]
    ) -> FieldResult:
        """Fuzzy name comparison to handle OCR variations and initials."""
        a = self._normalize_text(aadhaar_name)
        b = self._normalize_text(doc_name)

        if not a or not b:
            return FieldResult(
                "name", aadhaar_name, doc_name, False, 0.0,
                note="Name missing in one or both documents.",
            )

        similarity = self._string_similarity(a, b)
        match = similarity >= self.NAME_SIMILARITY_THRESHOLD

        if match:
            note = f"Name matches ({similarity:.0%} similarity)."
        elif similarity > 0.5:
            note = f"Partial name match ({similarity:.0%}). Possible abbreviation or OCR error."
        else:
            note = f"Name mismatch ({similarity:.0%} similarity)."

        logger.debug(f"Name: '{a}' vs '{b}' → similarity={similarity:.3f}, match={match}")
        return FieldResult("name", aadhaar_name, doc_name, match, similarity, note)

    def _compare_dob(
        self, aadhaar_dob: Optional[str], doc_dob: Optional[str]
    ) -> FieldResult:
        """Exact date comparison after normalization."""
        a = self._normalize_date(aadhaar_dob)
        b = self._normalize_date(doc_dob)

        if not a or not b:
            return FieldResult(
                "dob", aadhaar_dob, doc_dob, False, 0.0,
                note="DOB missing in one or both documents.",
            )

        match = a == b
        similarity = 1.0 if match else self._string_similarity(a, b)
        note = "DOB matches." if match else f"DOB mismatch: Aadhaar={a}, Document={b}"
        logger.debug(f"DOB: '{a}' vs '{b}' → match={match}")
        return FieldResult("dob", aadhaar_dob, doc_dob, match, similarity, note)

    def _compare_gender(
        self, aadhaar_gender: Optional[str], doc_gender: Optional[str]
    ) -> FieldResult:
        """Exact gender comparison after normalization."""
        a = (aadhaar_gender or "").strip().lower()
        b = (doc_gender or "").strip().lower()

        if not a or not b:
            return FieldResult(
                "gender", aadhaar_gender, doc_gender, False, 0.0,
                note="Gender missing in one or both documents.",
            )

        match = a == b
        note = (
            "Gender matches."
            if match
            else f"Gender mismatch: Aadhaar={aadhaar_gender}, Document={doc_gender}"
        )
        logger.debug(f"Gender: '{a}' vs '{b}' → match={match}")
        return FieldResult(
            "gender", aadhaar_gender, doc_gender, match, 1.0 if match else 0.0, note
        )

    def _compare_mobile(
        self, aadhaar_mobile: Optional[str], doc_mobile: Optional[str]
    ) -> FieldResult:
        """Exact mobile number comparison after stripping country code."""
        a = self._normalize_mobile(aadhaar_mobile)
        b = self._normalize_mobile(doc_mobile)

        if not a or not b:
            return FieldResult(
                "mobile_number", aadhaar_mobile, doc_mobile, False, 0.0,
                note="Mobile number missing in one or both documents.",
            )

        match = a == b
        note = (
            "Mobile number matches."
            if match
            else f"Mobile mismatch: Aadhaar={a}, Document={b}"
        )
        logger.debug(f"Mobile: '{a}' vs '{b}' → match={match}")
        return FieldResult(
            "mobile_number", aadhaar_mobile, doc_mobile,
            match, 1.0 if match else 0.0, note,
        )

    # --- Main verification ---

    def verify(
        self, aadhaar: AadhaarData, document: DocumentData
    ) -> VerificationReport:
        """
        Compare Aadhaar data against another document and return a report.

        Args:
            aadhaar:   Extracted AadhaarData.
            document:  Extracted DocumentData.

        Returns:
            VerificationReport with field-by-field results for
            name, dob, gender, and mobile_number.
        """
        logger.info("Starting document verification against Aadhaar.")

        comparisons: list[FieldResult] = [
            self._compare_name(aadhaar.name, document.name),
            self._compare_dob(aadhaar.dob, document.dob),
            self._compare_gender(aadhaar.gender, document.gender),
            self._compare_mobile(aadhaar.mobile_number, document.mobile_number),
        ]

        matched = []
        mismatched = []
        skipped = []
        weighted_score = 0.0
        total_weight = 0.0

        for result in comparisons:
            weight = self.FIELD_WEIGHTS.get(result.field_name, 0.1)
            both_missing = not result.aadhaar_value and not result.document_value
            one_missing = not result.aadhaar_value or not result.document_value

            if both_missing:
                skipped.append(result.field_name)
                logger.debug(f"Skipping '{result.field_name}': missing in both.")
                continue

            if one_missing:
                skipped.append(result.field_name)
                total_weight += weight  # Penalize — field couldn't be verified
                logger.debug(f"Skipping '{result.field_name}': missing in one document.")
                continue

            total_weight += weight
            weighted_score += result.similarity * weight

            if result.match:
                matched.append(result.field_name)
            else:
                mismatched.append(result.field_name)

        overall_score = (
            round(weighted_score / total_weight, 3) if total_weight > 0 else 0.0
        )
        is_valid = overall_score >= self.validity_threshold

        # Hard rule: severe name mismatch always invalidates the document
        name_result = next((r for r in comparisons if r.field_name == "name"), None)
        if name_result and not name_result.match and name_result.similarity < 0.5:
            is_valid = False
            logger.warning("Hard mismatch on name — marking document as invalid.")

        summary = (
            f"Document is {'VALID' if is_valid else 'INVALID'}. "
            f"Overall score: {overall_score:.0%}. "
            f"Matched: {matched}. "
            f"Mismatched: {mismatched}. "
            f"Skipped: {skipped}."
        )

        logger.info(summary)

        return VerificationReport(
            is_valid=is_valid,
            overall_score=overall_score,
            fields=comparisons,
            matched_fields=matched,
            mismatched_fields=mismatched,
            skipped_fields=skipped,
            summary=summary,
        )