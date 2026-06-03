from typing import List, Dict, Any, Type
from openai import AsyncOpenAI


class OpenAIModel:
    """
    Base class for OpenAI-based LLM components.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        api_model = model or self.model
        api_temp = temperature if temperature is not None else self.temperature
        api_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        params: Dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            **kwargs,
        }

        params["temperature"] = api_temp
        params["max_tokens"] = api_max_tokens

        if response_format is not None:
            params["response_format"] = response_format
            return await self.client.beta.chat.completions.parse(**params)

        return await self.client.chat.completions.create(**params)