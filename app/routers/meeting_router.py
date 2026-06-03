from fastapi import APIRouter, Depends, File, UploadFile, status, Form
from app.schemas.api_response_schema import ApiResponse
from app.schemas.meeting_schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse
)
from app.services.meeting_service import MeetingService
from app.core.config.dependencies import get_meeting_service

router = APIRouter(prefix="/api", tags=["Meetings"])

@router.post(
    "/analyze/file",
    response_model=ApiResponse[AnalyzeResponse],
    status_code=status.HTTP_200_OK,
    summary="Analyze uploaded meeting transcript file",
    description=(
        "Ingests an uploaded plain text file transcript, summarizes it, "
        "identifies Action Items categorized using the Eisenhower Matrix via OpenAI Structured Outputs, "
        "broadcasts the details to a Slack channel, and returns the parsed contents."
    )
)
async def analyze_file(
    file: UploadFile = File(
        ...,
        description="The plain text (.txt) file containing the raw meeting notes/transcript."
    ),
    context: str = Form(
        default="",
        description="Optional viewer context (e.g. 'I am an HR manager'). Tailors the summary to that perspective."
    ),
    service: MeetingService = Depends(get_meeting_service)
) -> ApiResponse[AnalyzeResponse]:
    """
    Endpoint to process meeting notes uploaded directly as a plain text file.
    """
    return await service.process_analysis_file(file, context)

@router.post(
    "/chat",
    response_model=ApiResponse[ChatResponse],
    status_code=status.HTTP_200_OK,
    summary="Interactive Q&A on meeting notes transcript",
    description=(
        "Allows users to ask questions about the meeting and receive direct answers. "
        "Maintains conversation history for follow-up questions."
    )
)
async def chat(
    request: ChatRequest,
    service: MeetingService = Depends(get_meeting_service)
) -> ApiResponse[ChatResponse]:
    """
    Endpoint for context-bound conversational QA about the transcript.
    """
    # The request object now contains request.chat_history automatically!
    return await service.process_chat(request)