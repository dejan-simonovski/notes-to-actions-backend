import httpx
from fastapi import UploadFile
from app.exceptions.meeting_exception import DocumentExtractionError

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Safely reads multi-part uploaded file bytes and decodes to a standard string.
    Tries multiple decodings (utf-8, utf-8-sig, latin-1) and falls back to utf-8
    with replacement characters to avoid decoding failures.
    """
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise DocumentExtractionError(
            f"Failed to read raw file bytes: {str(e)}"
        )
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            raise DocumentExtractionError(
            f"Uploaded file encoding error: {str(e)}"
        )

    try:
        return content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise DocumentExtractionError(
            f"Uploaded file encoding error: {str(e)}"
        )

