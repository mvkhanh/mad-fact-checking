import os
import json
import nltk
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from app.core.config import settings
from app.services import BM25Retriever, Reranker
from app.services.web_retrieval import WebSearchRetriever

def download_nltk_data(package_name, download_dir='nltk_data'):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(download_dir)
    
    try:
        # Try to find the resource
        nltk.data.find(f'tokenizers/{package_name}')
        print(f"Package '{package_name}' is already downloaded")
    except LookupError:
        # If resource isn't found, download it
        print(f"Downloading {package_name}...")
        nltk.download(package_name, download_dir=download_dir)
        print(f"Successfully downloaded {package_name}")

download_nltk_data('punkt')
download_nltk_data('punkt_tab')

class RetrievalInput(BaseModel):
    fact: str = Field(description="")
    queries: List[str] = Field(description="Queries generated for evidence retrieval.")
    idx: str = Field(description="The ID of the claim to locate the correct knowledge store file.")
    excluded_sentences: Optional[List[str]] = Field(default=[], description="Sentences need to skip")

_kb_reranker = None
_web_reranker = None
_kb_retriever = BM25Retriever(knowledge_store_path=settings.KNOWLEDGE_STORE_DIR, top_k=settings.TOP_K_RETRIEVAL, tavily_api_key=settings.TAVILY_API_KEY)
_web_retriever = WebSearchRetriever(
    top_k=settings.TOP_K_RETRIEVAL,
    max_results=settings.DEMO_TAVILY_MAX_RESULTS,
    search_depth=settings.DEMO_TAVILY_SEARCH_DEPTH,
    tavily_api_key=settings.TAVILY_API_KEY,
    cache_dir=settings.DEMO_CACHE_DIR,
)

def _get_reranker(use_web: bool) -> Reranker:
    global _kb_reranker, _web_reranker
    if use_web:
        if _web_reranker is None:
            _web_reranker = Reranker(model_name=settings.DEMO_EMBEDDING_MODEL, top_k=settings.TOP_K_RERANKING, pooling="mean")
        return _web_reranker
    else:
        if _kb_reranker is None:
            _kb_reranker = Reranker(model_name=settings.EMBEDDING_MODEL, top_k=settings.TOP_K_RERANKING)
        return _kb_reranker

class EvidenceRetrievalTool(BaseTool):
    name: str = "evidence_retrieval_tool"
    description: str = "Retrieves and reranks evidence using BM25 and Embedding (knowledge store or live web search)."
    args_schema: Type[BaseModel] = RetrievalInput

    def _run(self, fact: str, queries: List[str], idx: str, excluded_sentences: List[str] = []) -> str:
        use_web = not idx
        retriever = _web_retriever if use_web else _kb_retriever
        reranker = _get_reranker(use_web)
        top_k_sentence_urls = retriever.query(fact, queries, idx, excluded_sentences)
        if not top_k_sentence_urls:
            return "No evidence found in the knowledge store for this query."
        top_k_sentence_urls = reranker.rerank(fact, queries, top_k_sentence_urls, idx)
        formatted_result = json.dumps(top_k_sentence_urls, ensure_ascii=False, indent=2)
        return formatted_result

evidence_retrieval_tool = EvidenceRetrievalTool()