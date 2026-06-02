import asyncio
from typing import Optional, List
from app.llm.base.openai_model import OpenAIModel
from app.exceptions.meeting_exception import AiServiceError

class SummarizationModel(OpenAIModel):
    """
    Model class specialized in summarizing long documents using a Map-Reduce approach.

    The map_prompt and reduce_prompt are set once at construction time, so the caller
    never needs to pass prompts per-call. The summarize() method is the only public
    entry point; recursive calls reuse the same instance prompts without re-summarizing
    already-reduced content.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        map_prompt: Optional[str] = None,
        reduce_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._map_prompt = map_prompt 
        self._reduce_prompt = reduce_prompt

    def _split_text(self, text: str, max_chars: int = 8000, overlap: int = 1000) -> List[str]:
        """
        Splits text into chunks of at most max_chars characters, with a soft overlap.
        Prefers splitting at paragraph breaks, then newlines, then spaces.
        """
        if len(text) <= max_chars:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + max_chars

            if end < len(text):
                search_start = max(start, end - 500)
                split_idx = text.rfind("\n\n", search_start, end)
                if split_idx == -1:
                    split_idx = text.rfind("\n", search_start, end)
                if split_idx == -1:
                    split_idx = text.rfind(" ", search_start, end)
                if split_idx != -1 and split_idx > start:
                    end = split_idx + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            next_start = end - overlap
            
            if next_start <= start:
                next_start = start + 1
            start = next_start

            if start >= len(text):
                break

        return chunks

    async def _summarize_chunk(self, text: str, prompt: str) -> str:
        """Summarizes a single text chunk using the given prompt template."""
        messages = [
            {"role": "system", "content": "You are a helpful, professional summarization assistant."},
            {"role": "user", "content": prompt.format(text=text)},
        ]
        try:
            response = await self.create_chat_completion(messages=messages)
            content = response.choices[0].message.content
            if not content:
                raise AiServiceError("OpenAI returned an empty response during chunk summarization.")
            return content.strip()
        except AiServiceError:
            raise
        except Exception as e:
            raise AiServiceError(f"Error during chunk summarization: {str(e)}")

    async def summarize(
        self,
        document_content: str,
        chunk_size: int = 8000,
        chunk_overlap: int = 1000,
    ) -> str:
        """
        Summarizes document_content using a Map-Reduce approach.

        - Map:    each chunk is summarized with self._map_prompt in parallel.
        - Reduce: the combined chunk summaries are collapsed with self._reduce_prompt.

        If the combined summaries still exceed chunk_size, the method recurses — but
        it passes the summaries directly to the reduce step, never re-mapping them,
        so content is never summarized twice.

        Args:
            document_content: Raw text to summarize.
            chunk_size:        Max characters per chunk. Defaults to 8000.
            chunk_overlap:     Overlap between adjacent chunks. Defaults to 1000.

        Returns:
            Final summarized string.
        """
        if not document_content.strip():
            return ""

        chunks = self._split_text(document_content, max_chars=chunk_size, overlap=chunk_overlap)

        if len(chunks) == 1:
            return await self._summarize_chunk(chunks[0], prompt=self._reduce_prompt)

        map_tasks = [self._summarize_chunk(chunk, prompt=self._map_prompt) for chunk in chunks]
        chunk_summaries: List[str] = list(await asyncio.gather(*map_tasks))

        combined = "\n\n".join(chunk_summaries)

        if len(combined) > chunk_size:
            return await self._reduce_summaries(combined, chunk_size, chunk_overlap)

        return await self._summarize_chunk(combined, prompt=self._reduce_prompt)

    async def _reduce_summaries(
        self,
        combined: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        """
        Recursively collapses an oversized block of already-mapped summaries using
        only the reduce prompt — never the map prompt — so nothing is summarized twice.
        """
        chunks = self._split_text(combined, max_chars=chunk_size, overlap=chunk_overlap)

        if len(chunks) == 1:
            return await self._summarize_chunk(chunks[0], prompt=self._reduce_prompt)

        reduce_tasks = [self._summarize_chunk(chunk, prompt=self._reduce_prompt) for chunk in chunks]
        reduced: List[str] = list(await asyncio.gather(*reduce_tasks))

        next_combined = "\n\n".join(reduced)

        if len(next_combined) > chunk_size:
            return await self._reduce_summaries(next_combined, chunk_size, chunk_overlap)

        return await self._summarize_chunk(next_combined, prompt=self._reduce_prompt)