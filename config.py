import os
import ssl
import urllib3
import logging
from models import Settings
from dotenv import load_dotenv

# Disabling the ssl certification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
os.environ["NO_PROXY"] = "generativelanguage.googleapis.com"

# loading the env file
load_dotenv()

logger = logging.getLogger(__name__)

# Getting the evn variables
def load_settings() -> Settings:
    return Settings(
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
        DEBUG=os.getenv("DEBUG", "false").lower() == "true",
        VALIDITY_THRESHOLD=float(os.getenv("VALIDITY_THRESHOLD", "0.75")),
    )


settings = load_settings()