from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.schemas.api_response_schema import ApiResponse

class MeetingException(Exception):
    """
    Base exception class for all custom errors in the Meeting API.
    """
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class DocumentExtractionError(MeetingException):
    """
    Exception raised when reading text from a URL or uploaded file fails.
    """
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

class AiServiceError(MeetingException):
    """
    Exception raised when calls to the OpenAI service fail or cannot be parsed.
    """
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY
        )

async def meeting_exception_handler(request: Request, exc: MeetingException) -> JSONResponse:
    """
    Global FastAPI exception handler that intercepts MeetingExceptions
    and wraps them in a standard ApiResponse structure.
    """
    response_body = ApiResponse[None](
        success=False,
        message=exc.message,
        data=None
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_body.model_dump()
    )
