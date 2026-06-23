"""LLM service for managing OpenAI inference."""
import json
from typing import List, Optional, Any

from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings

class OpenAIService:
    def __init__(
        self, 
        model_name: str = "gpt-4.1-mini",
    ):
        print(f"Initializing OpenAI Engine for model: {model_name}")
        self.model_name = model_name

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.OPENAI_API_KEY, 
            max_tokens=7168,
            max_retries=2,
        )
        print(f"OpenAI Engine Initialization complete.")

    async def call(
        self,
        messages: List[BaseMessage],
        pydantic_model: Optional[Any] = None,
        **kwargs,
    ) -> BaseMessage:
        """
        Call OpenAI LLM with optional Structured Outputs.
        """
        if pydantic_model:
            structured_llm = self.llm.with_structured_output(pydantic_model, include_raw=True)
            response = await structured_llm.ainvoke(messages, **kwargs)
            parsed_obj = response["parsed"]
            json_str = parsed_obj.model_dump_json()
            clean_ai_message = AIMessage(content=json_str)
            return clean_ai_message
            
        else:
            response = await self.llm.ainvoke(messages, **kwargs)
            return response

llm_service = OpenAIService(
    model_name=getattr(settings, "DEFAULT_LLM_MODEL", "gpt-4.1-mini")
)