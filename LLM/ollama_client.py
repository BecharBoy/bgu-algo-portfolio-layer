from __future__ import annotations

import json
import os
from typing import Any, TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    def __init__(self) -> None:
        host = os.environ.get("OLLAMA_HOST")
        model = os.environ.get("OLLAMA_MODEL")
        if not host or not model:
            raise RuntimeError("OLLAMA_HOST and OLLAMA_MODEL must be configured")
        self.model_name = model
        self.client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=httpx.Timeout(300),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def structured(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[T],
        max_tokens: int = 2000,
    ) -> T:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model_name,
                "stream": False,
                "format": response_model.model_json_schema(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, default=str),
                    },
                ],
                "options": {
                    "temperature": 0,
                    "top_p": 0.1,
                    "seed": 42,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response is missing message.content")
        return response_model.model_validate_json(content)

