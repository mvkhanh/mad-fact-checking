"""Application configuration management.

This module handles environment-specific configuration loading, parsing, and management
for the application. It includes environment detection, .env file loading, and
configuration value parsing.
"""

import os

from dotenv import load_dotenv
load_dotenv()


class Settings:
    """Application settings without using pydantic."""

    def __init__(self):
        """Initialize application settings from environment variables.

        Loads and sets all configuration values from environment variables,
        with appropriate defaults for each setting. Also applies
        environment-specific overrides based on the current environment.
        """
        # LangGraph Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
        self.KNOWLEDGE_STORE_DIR = os.getenv("KNOWLEDGE_STORE_DIR", "../../knowledge_store/dev/")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 1000))
        self.TOP_K_RERANKING = int(os.getenv("TOP_K_RERANKING", 10))
        self.MAX_ROUND_DEBATE = int(os.getenv("MAX_ROUND_DEBATE", 3))
        self.DEMO_CACHE_DIR = os.getenv("DEMO_CACHE_DIR", "tmp_web_cache")
        self.DEMO_EMBEDDING_MODEL = os.getenv("DEMO_EMBEDDING_MODEL", "dangvantuan/vietnamese-document-embedding")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
        self.DEMO_TAVILY_MAX_RESULTS = int(os.getenv("DEMO_TAVILY_MAX_RESULTS", "7"))
        self.DEMO_TAVILY_SEARCH_DEPTH = os.getenv("DEMO_TAVILY_SEARCH_DEPTH", "advanced")
        
# Create settings instance
settings = Settings()
