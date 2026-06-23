import json
import os
import re
import numpy as np
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
from pathlib import Path

class BM25Retriever:
    def __init__(self, knowledge_store_path, top_k, tavily_api_key: str = ""):
        self.knowledge_store_path = knowledge_store_path
        self.top_k = top_k
        self.tavily_api_key = tavily_api_key

    def _get_cached_or_scrape(self, idx, split, missing_urls):
        tmp_dir = Path("tmp_knowledge_store") / split
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tmp_dir / f"{idx}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if not self.tavily_api_key:
            return {}

        print(f"[*] Claim {idx} ({split}): Extracting {len(missing_urls)} missing URLs via Tavily...")
        from tavily import TavilyClient
        client = TavilyClient(api_key=self.tavily_api_key)
        try:
            results = client.extract(urls=missing_urls).get("results", [])
        except Exception as e:
            print(f"  [!] Tavily extract error: {e}")
            results = []

        scraped_data = {}
        for r in results:
            url = r.get("url", "")
            content = r.get("raw_content", "")
            if url and content:
                flat_text = re.sub(r'\s+', ' ', content).strip()
                raw_sentences = nltk.sent_tokenize(flat_text)
                scraped_data[url] = [
                    s.strip() for s in raw_sentences
                    if len(s.strip()) > 15 and re.search(r'[a-zA-Z]', s)
                ]
            elif url:
                scraped_data[url] = []

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, ensure_ascii=False, indent=2)

        return scraped_data

    def combine_all_sentences(self, knowledge_file, idx, split, exclude_sentences=None):
        items = []
        with open(knowledge_file, "r", encoding="utf-8") as json_file:
            for line in json_file:
                items.append(json.loads(line))
        
        if split in ['dev', 'test']:
            urls_to_scrape = [
                data.get("url") for data in items[:12] 
                if len(data.get("url2text", [])) == 0
            ]
                    
            if urls_to_scrape:
                scraped_data = self._get_cached_or_scrape(idx, split, urls_to_scrape)
                for data in items[:12]:
                    url = data.get("url")
                    if url in scraped_data and scraped_data[url]:
                        data["url2text"] = scraped_data[url]

        chunks, urls = [], []
        exclude_set = set(exclude_sentences) if exclude_sentences else set()
        
        for data in items:
            url_sentences = data.get("url2text", [])
            n = len(url_sentences)
            for j in range(0, n, 3):
                parts = [url_sentences[k].strip() for k in range(j, min(j+5, n)) if url_sentences[k].strip()]
                chunk = " ".join(parts)
                chunk = re.sub(r'\s+', ' ', chunk).strip()

                if chunk and chunk not in exclude_set:
                    chunks.append(chunk)
                    urls.append(data.get("url"))
                    
        return chunks, urls, len(items)

    def remove_duplicates(self, sentences, urls):
        df = pd.DataFrame({"document_in_sentences": sentences, "sentence_urls": urls})
        df['sentences'] = df['document_in_sentences'].str.strip().str.lower()
        df = df.drop_duplicates(subset="sentences").reset_index()
        return df['document_in_sentences'].tolist(), df['sentence_urls'].tolist()

    def retrieve_top_k_sentences(self, query, document, urls, max_sentences_per_url=2):
        if not document: return [], []
            
        tokenized_docs = [nltk.word_tokenize(doc) for doc in document[:20000]]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(query))
        sorted_indices = np.argsort(scores)[::-1]
        
        top_k_sentences, top_k_urls, url_counts = [], [], {}
        
        for idx in sorted_indices:
            if len(top_k_sentences) >= self.top_k: break
            current_sentence, current_url = document[idx], urls[idx]
            
            if url_counts.setdefault(current_url, 0) < max_sentences_per_url:
                top_k_sentences.append(current_sentence)
                top_k_urls.append(current_url)
                url_counts[current_url] += 1
                
        return top_k_sentences, top_k_urls

    def query(self, fact, queries, idx, exclude_sentences=None):
        try:
            split = 'dev' if 'dev' in self.knowledge_store_path else 'test'
            knowledge_file_path = os.path.join(self.knowledge_store_path, f"{idx}.json")
            document_in_sentences, sentence_urls, num_urls_this_claim = self.combine_all_sentences(
                knowledge_file_path, idx, split, exclude_sentences
            )
            
            if not document_in_sentences: return []
                
            document_in_sentences, sentence_urls = self.remove_duplicates(document_in_sentences, sentence_urls)
            query_str = fact + " " + " ".join(queries)
            
            top_k_sentences, top_k_urls = self.retrieve_top_k_sentences(query_str, document_in_sentences, sentence_urls)
            
            return [{"sentence": re.sub(r'^Pingback:\s*', '', s, flags=re.IGNORECASE).strip(), "url": u} 
                    for s, u in zip(top_k_sentences, top_k_urls)] 
        except Exception as e:
            print(f"Error processing example {idx} in SparseRetriever: {str(e)}")
            return None