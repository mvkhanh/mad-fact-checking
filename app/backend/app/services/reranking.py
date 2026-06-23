from tqdm import tqdm
import torch
import gc
from transformers import AutoModel, AutoTokenizer
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = 'auto', top_k: int = 10, batch_size: int = 16, pooling: str = 'cls'):
        self.model_name = model_name
        
        # 1. Device Setup
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Reranker (Bi-Encoder) using device: {self.device}")
        
        if self.device == 'cuda':
            self.dtype = torch.bfloat16
        elif self.device == 'mps':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        # torch.compile gives 10-30% speedup after the first (compilation) call
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.top_k = top_k
        self.batch_size = batch_size
        self.pooling = pooling  # 'cls' or 'mean'

    def encode(self, texts: list[str], instruction: str = "") -> np.ndarray:
        """Generate embeddings for doc list"""
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]
            
        all_embeddings = []
        
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding", leave=False):
                batch = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                if self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=self.dtype):
                        outputs = self.model(**inputs)
                elif self.device == 'mps':
                    with torch.autocast(device_type='mps', dtype=self.dtype):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                if self.pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
                
                if i > 0 and i % (self.batch_size * 4) == 0:
                    if self.device == 'cuda': torch.cuda.empty_cache()
                    elif self.device == 'mps': torch.mps.empty_cache()
                    gc.collect()

        return np.vstack(all_embeddings)

    def remove_special_chars_except_spaces(self, text):
        return re.sub(r'[^\w\s]+', '', text)
    
    def preprocess_sentences(self, sentence1, sentence2):
        vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0][1]

    def select_top_k(self, claim, results):
        '''
        Refine to top-k
        '''
        dup_check = set()
        top_k_sentences_urls = []
        
        i = 0
        claim = self.remove_special_chars_except_spaces(claim).lower()
        while len(top_k_sentences_urls) < self.top_k and i < len(results):
            sentence = self.remove_special_chars_except_spaces(results[i]['sentence']).lower()
            
            if sentence not in dup_check:
                if self.preprocess_sentences(claim, sentence) > 0.97:
                    dup_check.add(sentence)
                    i += 1
                    continue
                
                if claim in sentence:
                    if len(claim) / len(sentence) > 0.92:
                        dup_check.add(sentence)
                        i += 1
                        continue 
                
                top_k_sentences_urls.append({
                    'sentence': results[i]['sentence'],
                    'url': results[i]['url']}
                )
            i += 1
            
        return top_k_sentences_urls

    def rerank(self, fact, queries, top_sentences_urls, idx):
        print(f'Start rerank for claim: {idx}')
        
        combined_query = fact + " " + " ".join([q for q in queries if len(q.strip()) > 0])
        sentences = [sent['sentence'] for sent in top_sentences_urls]
        
        if not sentences:
            return []

        try:
            st = time.time()
            query_instruction = "Represent this sentence for searching relevant passages: "
            query_emb = self.encode([combined_query], instruction=query_instruction)
            doc_embs = self.encode(sentences)
            scores = np.dot(doc_embs, query_emb.T).flatten()
            top_k_idx = np.argsort(scores)[::-1]
            results = [top_sentences_urls[i] for i in top_k_idx]
            final_top_k_sentences_urls = self.select_top_k(fact, results)
            print(f"Top {len(final_top_k_sentences_urls)} retrieved via Bi-Encoder. Time elapsed: {time.time() - st:.2f}s")
            return final_top_k_sentences_urls
            
        except Exception as e:
            print(f"Error processing claim {idx} in Bi-Encoder Reranker: {e}")
            return []