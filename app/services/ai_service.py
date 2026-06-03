import os
from typing import Any, Dict
from openai import AsyncOpenAI
from app.schemas.meeting_schema import AnalyzeResponse
from app.exceptions.meeting_exception import AiServiceError
from app.core.config.openai_config import settings
from app.core.prompts import (
    load_system_instruction,
    load_map_prompt,
    load_reduce_prompt,
)

class AiService:
    """
    Service layer interacting with OpenAI's API.
    Handles structured transcription analyses and free-text question answering.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL
        self.model_temperature = settings.OPENAI_TEMPERATURE
        self.model_max_tokens = settings.OPENAI_MAX_TOKENS
        self._client = AsyncOpenAI(api_key=self._api_key)


    async def analyze_transcript(self, transcript: str, context: str = "") -> dict[str, Any]:
        """
        Calls OpenAI using Structured Outputs to generate an executive summary
        and extract action items prioritized via the Eisenhower Matrix.
        Uses the SummarizationModel to preprocess the transcript using a Map-Reduce approach.
        If a viewer context is provided, the summary is tailored to that perspective.
        """
        if not self._api_key:
            raise AiServiceError(
                "OpenAI API Key is not configured. Please set the OPENAI_API_KEY environment variable."
            )
        try:
            from app.llm import SummarizationModel, AnalysisModel

            summarizer = SummarizationModel(
                api_key=self._api_key,
                model=self.model_name,
                temperature=self.model_temperature,
                max_tokens=self.model_max_tokens,
                map_prompt=load_map_prompt(),
                reduce_prompt=load_reduce_prompt(context=context),
            )
            analyzer = AnalysisModel(
                api_key=self._api_key,
                model=self.model_name,
                temperature=self.model_temperature,
                max_tokens=self.model_max_tokens,
                system_instruction=load_system_instruction(context=context),
            )

            summarized_content = await summarizer.summarize(transcript)
            response_data = await analyzer.analyze(
            content=f"Meeting Transcript Summary:\n{summarized_content}",
            response_format=AnalyzeResponse,
            timeout=10.0,
            )
            response_dict = response_data.model_dump()
            response_dict["transcript"] = transcript
            return response_dict

        except AiServiceError:
            raise
        except Exception as e:
            raise AiServiceError(f"Error during transcript analysis: {str(e)}")


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
