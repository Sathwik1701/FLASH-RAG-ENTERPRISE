import hashlib
from typing import Tuple

def calculate_sha256(file_bytes: bytes) -> str:
    """Computes a SHA-256 hash of the incoming binary stream for deduplication."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def validate_pdf_file(filename: str, content_type: str, file_bytes: bytes) -> Tuple[bool, str]:
    """Validates whether the uploaded file is a valid PDF."""
    if not filename.lower().endswith(".pdf"):
        return False, "File extension must be .pdf"
    
    if content_type not in ["application/pdf", "application/x-pdf", "application/octet-stream"]:
        return False, f"Invalid Content-Type: {content_type}. Must be application/pdf"
    
    # Check PDF magic bytes (%PDF-)
    if len(file_bytes) < 5 or not file_bytes.startswith(b"%PDF-"):
        return False, "File is missing standard PDF magic bytes (%PDF-)"
    
    return True, "Valid PDF"
