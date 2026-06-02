from typing import Optional, Type, TypeVar, Any
from pydantic import BaseModel
from app.llm.base.openai_model import OpenAIModel
from app.exceptions.meeting_exception import AiServiceError

T = TypeVar("T", bound=BaseModel)


class AnalysisModel(OpenAIModel):
    """
    Model class specialized in structured output extraction.
    Wraps OpenAIModel.create_chat_completion with response_format support,
    so the caller never touches the raw client directly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._system_instruction = system_instruction

    async def analyze(
        self,
        content: str,
        response_format: Type[T],
        timeout: float = 10.0,
    ) -> T:
        """
        Sends content to the model and parses the response into response_format.

        Args:
            content:         The user message to analyze.
            response_format: A Pydantic model class for structured output parsing.
            timeout:         Request timeout in seconds.

        Returns:
            A parsed instance of response_format.
        """
        messages = [
            {"role": "system", "content": self._system_instruction},
            {"role": "user", "content": content},
        ]

        try:
            completion = await self.create_chat_completion(
                messages=messages,
                response_format=response_format,
                timeout=timeout,
            )
            result = completion.choices[0].message.parsed
            
            if not result:
                raise AiServiceError("OpenAI returned an empty structured output.")
            return result
        except AiServiceError:
            raise
        except Exception as e:
            raise AiServiceError(f"Error during structured analysis: {str(e)}")