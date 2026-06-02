import os
from typing import Any, Dict
from openai import AsyncOpenAI
from app.schemas.meeting_schema import AnalyzeResponse
from app.exceptions.meeting_exception import AiServiceError

class AiService:
    """
    Service layer interacting with OpenAI's API.
    Handles structured transcription analyses and free-text question answering.
    """
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=self._api_key)

    async def analyze_transcript(self, transcript: str) -> dict[str, Any]:
        """
        Calls OpenAI using Structured Outputs to generate an executive summary
        and extract action items prioritized via the Eisenhower Matrix.
        """
        if not self._api_key or self._api_key == "mock-key-replace-this-to-test-live-openai":
            raise AiServiceError(
                "OpenAI API Key is not configured. Please set the OPENAI_API_KEY environment variable."
            )

        system_instruction = (
            "You are a stellar meeting productivity assistant.\n"
            "Analyze the meeting notes transcript and generate structured outputs strictly conforming to the JSON schema:\n"
            "1. Extract or determine a fitting, concise 'title' for the meeting.\n"
            "2. Synthesize an executive 'summary' capturing key updates, decisions made, and themes.\n"
            "3. Build a list of 'action_items'. For each action item:\n"
            "   - Extract a clear and active 'description'.\n"
            "   - Identify the clear 'assignee_name' (default to 'Unassigned' if not mentioned).\n"
            "   - Categorize the task priority ('priority') using the Eisenhower Matrix:\n"
            "     * 'urgent_important': High impact, high time-sensitivity.\n"
            "     * 'important_not_urgent': High impact, low time-sensitivity.\n"
            "     * 'urgent_not_important': Low impact, high time-sensitivity.\n"
            "     * 'low_priority': Low impact, low time-sensitivity.\n"
            "   - Keep 'status' defaulting to 'to_do' as a placeholder.\n\n"
            "Be comprehensive and ensure that the response adheres strictly to the model schema."
        )

        try:
            completion = await self._client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Meeting Transcript:\n{transcript}"}
                ],
                response_format=AnalyzeResponse,
                timeout=60.0
            )

            response_data = completion.choices[0].message.parsed
            if not response_data:
                raise AiServiceError("OpenAI returned an empty structured output.")

            return response_data.model_dump()

        except Exception as e:
            raise AiServiceError(f"Error calling OpenAI parsing service: {str(e)}")

    async def chat_with_transcript(self, transcript: str, question: str) -> str:
        """
        Interactively answers a question about a meeting transcript.
        Sourced strictly from the transcript context.
        """
        if not self._api_key or self._api_key == "mock-key-replace-this-to-test-live-openai":
            raise AiServiceError(
                "OpenAI API Key is not configured. Please set the OPENAI_API_KEY environment variable."
            )

        system_instruction = (
            "You are an expert Q&A meeting assistant.\n"
            "You will be given a transcript and a user's question about the meeting.\n"
            "Answer the question accurately, concisely, and based exclusively on the context of the transcript.\n"
            "If the answer cannot be found in the transcript, politely reply that it is not discussed in the meeting notes."
        )

        user_content = (
            f"Here is the meeting transcript:\n{transcript}\n\n"
            f"User Question: {question}"
        )

        try:
            completion = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                timeout=30.0
            )

            answer = completion.choices[0].message.content
            if not answer:
                raise AiServiceError("OpenAI returned an empty chat completion response.")

            return answer.strip()

        except Exception as e:
            raise AiServiceError(f"Error calling OpenAI chat completion: {str(e)}")
