import json
import re
import nltk
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError(
        "tavily-python is required for web search mode. "
        "Install it with: pip install tavily-python"
    )


class WebSearchRetriever:
    """Evidence retriever that uses Tavily to search and retrieve pre-extracted page content.

    Drop-in replacement for BM25Retriever.query() in demo/live mode.
    Accepts the same arguments and returns the same List[{sentence, url}] format.
    """

    def __init__(
        self,
        top_k: int,
        max_results: int,
        tavily_api_key: str,
        cache_dir: str,
        search_depth: str = "advanced",
    ):
        self.top_k = top_k
        self.max_results = max_results
        self.search_depth = search_depth
        self.cache_dir = Path(cache_dir)
        self._client = TavilyClient(api_key=tavily_api_key)

    # ------------------------------------------------------------------
    # Public entry point — same signature as BM25Retriever.query()
    # ------------------------------------------------------------------

    def query(
        self,
        fact: str,
        queries: List[str],
        idx: str,
        exclude_sentences: Optional[List[str]] = None,
    ) -> List[Dict]:
        try:
            search_terms = list(dict.fromkeys([fact] + queries))  # deduped, order preserved

            # 1. Load cache; search Tavily only for URLs not already cached
            cached = self._load_cache(idx)
            new_results = self._tavily_search(search_terms, cached_urls=set(cached.keys()))

            # 2. Merge and persist
            merged = {**cached, **new_results}
            self._save_cache(idx, merged)

            # 3. Build sliding-window chunks
            exclude_set = set(exclude_sentences) if exclude_sentences else set()
            chunks, chunk_urls = self._build_corpus(merged, exclude_set)

            if not chunks:
                return []

            # 4. Deduplicate then BM25
            from app.services.retrieval import BM25Retriever
            _bm25 = BM25Retriever(knowledge_store_path="", top_k=self.top_k)
            chunks, chunk_urls = _bm25.remove_duplicates(chunks, chunk_urls)

            query_str = fact + " " + " ".join(queries)
            top_sentences, top_urls = _bm25.retrieve_top_k_sentences(query_str, chunks, chunk_urls)

            return [
                {"sentence": re.sub(r'^Pingback:\s*', '', s, flags=re.IGNORECASE).strip(), "url": u}
                for s, u in zip(top_sentences, top_urls)
            ]
        except Exception as e:
            print(f"[WebSearchRetriever] Error during query: {e}")
            return []

    # ------------------------------------------------------------------
    # Tavily search
    # ------------------------------------------------------------------

    def _tavily_search(self, search_terms: List[str], cached_urls: set) -> Dict[str, List[str]]:
        """Search Tavily for each term concurrently; return {url: [sentences]} for newly discovered URLs."""
        url_to_sentences: Dict[str, List[str]] = {}

        def _search_one(term: str):
            try:
                return self._client.search(
                    term[:400],
                    max_results=self.max_results,
                    search_depth=self.search_depth,
                    include_raw_content=True,
                ).get("results", [])
            except Exception as e:
                print(f"[WebSearchRetriever] Tavily error for '{term}': {e}")
                return []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(len(search_terms), 5)) as pool:
            futures = {pool.submit(_search_one, t): t for t in search_terms}
            for future in as_completed(futures):
                for r in future.result():
                    url = r.get("url", "")
                    if not url or url in cached_urls or url in url_to_sentences:
                        continue
                    content = r.get("raw_content") or r.get("content", "")
                    if content:
                        url_to_sentences[url] = self._split_sentences(content)

        print(f"[WebSearchRetriever] Tavily returned {len(url_to_sentences)} new URLs")
        return url_to_sentences

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self, idx: str) -> Dict[str, List[str]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{idx}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, idx: str, data: Dict[str, List[str]]) -> None:
        cache_file = self.cache_dir / f"{idx}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WebSearchRetriever] Failed to write cache: {e}")

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Normalize whitespace and tokenize into clean sentences."""
        flat = re.sub(r'\s+', ' ', text).strip()
        raw = nltk.sent_tokenize(flat)
        return [
            s.strip() for s in raw
            if len(s.strip()) > 15 and re.search(r'[a-zA-Z]', s)
        ]

    def _build_corpus(
        self,
        url_to_sentences: Dict[str, List[str]],
        exclude_set: set,
    ) -> Tuple[List[str], List[str]]:
        """Sliding window (5 sentences, stride 3) over each URL's sentence list."""
        chunks, urls = [], []
        for url, sentences in url_to_sentences.items():
            n = len(sentences)
            for j in range(0, n, 3):
                parts = [sentences[k].strip() for k in range(j, min(j + 5, n)) if sentences[k].strip()]
                chunk = re.sub(r'\s+', ' ', " ".join(parts)).strip()
                if chunk and chunk not in exclude_set:
                    chunks.append(chunk)
                    urls.append(url)
        return chunks, urls
