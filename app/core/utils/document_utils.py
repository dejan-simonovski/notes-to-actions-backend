import httpx
from fastapi import UploadFile
from app.exceptions.meeting_exception import DocumentExtractionError

async def extract_text_from_url(url: str) -> str:
    """
    Fetches a Google Doc text export URL asynchronously using httpx.AsyncClient.
    Assume URL format: https://docs.google.com/document/d/{id}/export?format=txt
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise DocumentExtractionError(
                    f"Failed to fetch document. Status code: {response.status_code}"
                )
            return response.text
    except httpx.RequestError as e:
        raise DocumentExtractionError(
            f"Network transport error while pulling text from URL: {str(e)}"
        )
    except DocumentExtractionError:
        raise
    except Exception as e:
        raise DocumentExtractionError(
            f"Unexpected error extracting text from URL: {str(e)}"
        )

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Safely reads multi-part uploaded file bytes and decodes to a standard string.
    """
    try:
        content_bytes = await file.read()
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DocumentExtractionError(
            f"Uploaded file encoding error. Only standard UTF-8 text is supported: {str(e)}"
        )
    except Exception as e:
        raise DocumentExtractionError(
            f"Failed to read raw file bytes: {str(e)}"
        )
