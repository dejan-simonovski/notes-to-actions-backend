import asyncio
import httpx
from fastapi import UploadFile
from app.schemas.api_response_schema import ApiResponse
from app.schemas.meeting_schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse
)
from app.services.ai_service import AiService
from app.core.config.meeting_config import MeetingConfig
from app.core.utils import document_utils

# In-memory store for global chatbot context
GLOBAL_MEETINGS_STORE = []

class MeetingService:
    """
    Orchestration layer coordinates text extraction, AI processing, and webhook triggers.
    """
    def __init__(self, ai_service: AiService, config: MeetingConfig):
        self._ai_service = ai_service
        self._config = config

    async def _send_slack_webhook(self, data: AnalyzeResponse) -> None:
        """
        Asynchronously sends a formatted markdown message of action items to Slack.
        Wrapped entirely in try-except block to make sure webhook failure never crashes the main thread.
        """
        webhook_url = self._config.SLACK_WEBHOOK_URL
        if not webhook_url or webhook_url == "https://hooks.slack.com/services/MOCK/URL":
            # Silently skip sending if mock URL is used
            return

        priority_emojis = {
            "urgent_important": "🔴 *Urgent & Important*",
            "important_not_urgent": "🟠 *Important but Not Urgent*",
            "urgent_not_important": "🟡 *Urgent but Not Important*",
            "low_priority": "⚪ *Low Priority*"
        }

        # Build markdown lines
        lines = [
            f"📅 *Meeting Summary & Action Items: {data.title}*",
            f"\n*Executive Summary:*",
            f"{data.summary}",
            f"\n*Action Items List:*"
        ]

        if not data.action_items:
            lines.append("_No action items identified._")
        else:
            for idx, item in enumerate(data.action_items, 1):
                priority_label = priority_emojis.get(item.priority, str(item.priority))
                lines.append(
                    f"{idx}. *{item.description}*\n"
                    f"   • Assignee: `{item.assignee_name}` | Status: `{item.status}` | Priority: {priority_label}"
                )

        payload = {"text": "\n".join(lines)}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook_url, json=payload)
                # Failures are absorbed silently to keep main logic 100% resilient
                if response.status_code >= 400:
                    pass
        except Exception:
            pass

    async def process_analysis_url(self, request: AnalyzeRequest) -> ApiResponse[AnalyzeResponse]:
        """
        Extracts transcript text from a URL, analyzes it using OpenAI structured output,
        triggers the Slack webhook asynchronously, and returns a generic response wrapper.
        """
        if not request.url:
            from app.exceptions.meeting_exception import DocumentExtractionError
            raise DocumentExtractionError("A valid URL must be provided in the request payload.")

        raw_text = await document_utils.extract_text_from_url(str(request.url))

        contextual_transcript = (
            f"Meeting Title Context: {request.title}\n"
            f"Meeting Date Context: {request.date}\n\n"
            f"Transcript Content:\n{raw_text}"
        )

        analysis_data = await self._ai_service.analyze_transcript(contextual_transcript)

        if request.title:
            analysis_data["title"] = request.title

        analyze_response = AnalyzeResponse(**analysis_data)

        asyncio.create_task(self._send_slack_webhook(analyze_response))

        # 4. Save to global store for chatbot context
        GLOBAL_MEETINGS_STORE.append({
            "title": request.title,
            "date": request.date,
            "summary": analyze_response.summary,
            "action_items": [item.model_dump() for item in analyze_response.action_items],
            "transcript": raw_text
        })

        return ApiResponse[AnalyzeResponse](
            success=True,
            message="Meeting transcript from URL successfully analyzed.",
            data=analyze_response
        )

    async def process_analysis_file(self, file: UploadFile, context: str = "") -> ApiResponse[AnalyzeResponse]:
        """
        Extracts transcript text from an uploaded file, analyzes it,
        posts to Slack in the background, and returns the response.
        """
        raw_text = await document_utils.extract_text_from_file(file)
        analysis_data = await self._ai_service.analyze_transcript(raw_text, context)
        analyze_response = AnalyzeResponse(**analysis_data)
        asyncio.create_task(self._send_slack_webhook(analyze_response))

        GLOBAL_MEETINGS_STORE.append({
            "title": analyze_response.title,
            "date": "Unknown Date",  # files don't receive date from the frontend currently
            "summary": analyze_response.summary,
            "action_items": [item.model_dump() for item in analyze_response.action_items],
            "transcript": raw_text
        })

        return ApiResponse[AnalyzeResponse](
            success=True,
            message="Uploaded transcript file successfully analyzed.",
            data=analyze_response
        )

    async def process_chat(self, request: ChatRequest) -> ApiResponse[ChatResponse]:
        """
        Interacts with the AI service to answer a specific question on a transcript,
        maintaining conversation history.
        """
        # Convert Pydantic models to dicts for the AI service
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history] if request.history else []
        
        answer_text = await self._ai_service.chat_with_transcript(
            transcript=request.transcript,
            question=request.question,
            history=history_dicts,
            title=request.title,
            date=request.date,
            summary=request.summary,
            action_items=[item.model_dump() for item in request.action_items] if request.action_items else None,
            global_meetings=GLOBAL_MEETINGS_STORE
        )

        chat_response = ChatResponse(answer=answer_text)

        return ApiResponse[ChatResponse](
            success=True,
            message="Question answered successfully.",
            data=chat_response
        )
