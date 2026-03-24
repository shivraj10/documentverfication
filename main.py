import asyncio
import logging
import sys


from config import settings

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from extractors.aadhaar_extractor import AadhaarExtractor, AadhaarData
from extractors.document_extractor import DocumentExtractor, DocumentData
from verifier.document_verifier import DocumentVerifier, VerificationReport


# Logging Setup
def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Aadhaar Verification API Starting Up ===")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    logger.info("Aadhaar OCR: Tesseract + OpenCV (local, no API)")
    logger.info("Document OCR: Gemini Vision 1.5 Flash (REST, SSL verify=False)")
    yield
    logger.info("=== Aadhaar Verification API Shutting Down ===")


# App Initialization
app = FastAPI(
    title="Aadhaar Document Verification API",
    description=(
        "Upload an Aadhaar card front image and any other identity document. "
        "The API extracts information from both and verifies whether they match."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency Injection
def get_aadhaar_extractor() -> AadhaarExtractor:
    return AadhaarExtractor(api_key=settings.GEMINI_API_KEY)


def get_document_extractor() -> DocumentExtractor:
    return DocumentExtractor(api_key=settings.GEMINI_API_KEY)


def get_verifier() -> DocumentVerifier:
    return DocumentVerifier(validity_threshold=settings.VALIDITY_THRESHOLD)


# Validators
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE_MB = 10

# Image validation for given parameters
async def validate_image(file: UploadFile) -> bytes:
    """Read and validate uploaded image file."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Rejected file with content type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP.",
        )
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        logger.warning(
            f"Rejected file: size {size_mb:.1f}MB exceeds limit {MAX_FILE_SIZE_MB}MB"
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB.",
        )
    logger.debug(
        f"Validated image: {file.filename}, {size_mb:.2f}MB, {file.content_type}"
    )
    return image_bytes


# Route for extract information from an aadhaar card
@app.post(
    "/extract/aadhaar",
    tags=["Extraction"],
    summary="Extract information from Aadhaar card front image",
    response_description="Extracted Aadhaar fields",
)
async def extract_aadhaar(
    file: UploadFile = File(..., description="Aadhaar card front image (JPEG/PNG)"),
    extractor: AadhaarExtractor = Depends(get_aadhaar_extractor),
):
    """
    Upload the **front side** of an Aadhaar card.
    Returns extracted: name, DOB, gender, mobile number, Aadhaar number.
    """
    logger.info(f"POST /extract/aadhaar — file: {file.filename}")

    # Validate the image
    image_bytes = await validate_image(file)

    # Extract the information from aadhaar card
    aadhaar_data: AadhaarData = await extractor.extract(image_bytes)

    if aadhaar_data.errors and not aadhaar_data.name:
        logger.error(f"Aadhaar extraction failed: {aadhaar_data.errors}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Failed to extract Aadhaar data.",
                "errors": aadhaar_data.errors,
            },
        )

    logger.info(
        f"Aadhaar extraction successful. Confidence: {aadhaar_data.extraction_confidence}"
    )
    return {
        "success": True,
        "data": aadhaar_data.to_dict(),
    }

# Route to extract information from document
@app.post(
    "/extract/document",
    tags=["Extraction"],
    summary="Extract information from any identity document",
    response_description="Extracted document fields",
)
async def extract_document(
    file: UploadFile = File(..., description="Identity document image (JPEG/PNG)"),
    extractor: DocumentExtractor = Depends(get_document_extractor),
):
    """
    Upload **any identity document** (PAN, Driving Licence, Passport, Voter ID, etc.).
    Returns extracted: document type, name, DOB, gender, mobile number, ID number.
    """
    logger.info(f"POST /extract/document — file: {file.filename}")

    # Validate the image
    image_bytes = await validate_image(file)

    # Extract information from the document
    doc_data: DocumentData = await extractor.extract(image_bytes)

    if doc_data.errors and not doc_data.name:
        logger.error(f"Document extraction failed: {doc_data.errors}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Failed to extract document data.",
                "errors": doc_data.errors,
            },
        )

    logger.info(
        f"Document extraction successful. Type: {doc_data.document_type}. "
        f"Confidence: {doc_data.extraction_confidence}"
    )
    return {
        "success": True,
        "data": doc_data.to_dict(),
    }

# Route to verify if the information from aadhaar is matching from the information in document
@app.post(
    "/verify",
    tags=["Verification"],
    summary="Verify a document against an Aadhaar card",
    response_description="Verification report with field-by-field results",
)
async def verify_document(
    aadhaar_file: UploadFile = File(..., description="Aadhaar card front image"),
    document_file: UploadFile = File(..., description="Other identity document image"),
    aadhaar_extractor: AadhaarExtractor = Depends(get_aadhaar_extractor),
    doc_extractor: DocumentExtractor = Depends(get_document_extractor),
    verifier: DocumentVerifier = Depends(get_verifier),
):
    """
    Upload **both** an Aadhaar card and another document.
    The API will:
    1. Extract information from both documents concurrently via Gemini Vision.
    2. Compare name, DOB, gender, and mobile number.
    3. Return a verification report with is_valid flag and field-by-field breakdown.
    """
    logger.info(
        f"POST /verify — aadhaar: {aadhaar_file.filename}, document: {document_file.filename}"
    )

    # Validate and read both files
    aadhaar_bytes, document_bytes = await asyncio.gather(
        validate_image(aadhaar_file),
        validate_image(document_file),
    )

    # Extract from both documents concurrently
    logger.info("Extracting Aadhaar and document data concurrently...")
    aadhaar_data, doc_data = await asyncio.gather(
        aadhaar_extractor.extract(aadhaar_bytes),
        doc_extractor.extract(document_bytes),
    )

    if not aadhaar_data.name and aadhaar_data.errors:
        logger.error(f"Aadhaar extraction failed during verify: {aadhaar_data.errors}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Failed to extract Aadhaar data.",
                "errors": aadhaar_data.errors,
            },
        )

    if not doc_data.name and doc_data.errors:
        logger.error(f"Document extraction failed during verify: {doc_data.errors}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Failed to extract document data.",
                "errors": doc_data.errors,
            },
        )

    # Verify the information
    logger.info("Running verification...")
    report: VerificationReport = verifier.verify(aadhaar_data, doc_data)

    logger.info(
        f"Verification complete — is_valid={report.is_valid}, score={report.overall_score}"
    )

    return {
        "success": True,
        "is_valid": report.is_valid,
        "overall_score": report.overall_score,
        "aadhaar_data": aadhaar_data.to_dict(),
        "document_data": doc_data.to_dict(),
        "verification_report": report.to_dict(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error.",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )