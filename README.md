# Aadhaar Document Verification API

A FastAPI service that extracts and verifies identity fields from Aadhaar cards and other Indian identity documents (PAN, Driving Licence, Passport, Voter ID) using Google Gemini Vision.

---

## Project Structure

```
documentverification/
├── .env
├── main.py
├── config.py
├── requirements.txt
├── extractors/
│   ├── __init__.py
│   ├── aadhar_extractor.py      # Gemini Vision — Aadhaar cards
│   └── document_extractor.py   # Gemini Vision — all other documents
└── verifier/
    ├── __init__.py
    └── document_verifier.py
```

---

## Prerequisites

- Python 3.10+
- A free Google Gemini API key — get one at https://aistudio.google.com/app/apikey

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd documentverification
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the `.env` file

Create a file named `.env` in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=false
VALIDITY_THRESHOLD=0.75
```

## Running the API

```bash
python main.py
```

The API will be available at: http://localhost:8000

Interactive API docs (Swagger UI): http://localhost:8000/docs

---
