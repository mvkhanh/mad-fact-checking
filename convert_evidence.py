import json
import csv
import re
import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load API key
load_dotenv('/Users/mvk/FinalProject/FEVER-8-Shared-Task/app/backend/.env')

INPUT_FILE = '/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/final_results.json'
OUTPUT_CSV = '/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/submission2.csv'

# LLM setup
llm = ChatOpenAI(
    model=os.getenv("OPEN_ROUTER_MODEL"),
    api_key=os.getenv("OPEN_ROUTER_API_KEY", ""),
    base_url=os.getenv("OPEN_ROUTER_BASE_URL"),
    temperature=0.0,
    max_tokens=2500
)

def _parse_retry_after(error: Exception) -> Optional[float]:
    """
    Cố gắng trích xuất retry_after_seconds từ error message của OpenRouter 429.
    Trả về số giây cần chờ, hoặc None nếu không parse được.
    """
    error_str = str(error)
    # Tìm 'retry_after_seconds': <number>
    match = re.search(r"'retry_after_seconds':\s*(\d+)", error_str)
    if match:
        return float(match.group(1))
    # Tìm 'Retry-After': <number>
    match = re.search(r"'Retry-After':\s*'(\d+)'", error_str)
    if match:
        return float(match.group(1))
    return None

async def generate_question(claim: str, verdict: str, justification: str, formatted_evidences: List[str], max_retries: int = 5) -> str:
    """Gọi LLM để sinh danh sách QA, có retry khi bị rate limit."""
    evidence_context = "\n".join(formatted_evidences)
    
    prompt = f"""You are an expert fact-checker making a dataset.
    You will be provided the original claim, the output verdict label, the justification to explain how evidence led to that label, and the original evidence lists (Sentences format).
    Your task is to create an evidence list in Question-Answer format. Extract and format only the evidences that contribute directly to the verdict label. Skip irrelevant parts in the justification that don't contribute to the label.
    The evidence list should look like a step by step reasoning that lead to verdict label.
    Example of correct output:
    "Where was the claim first published?
    It was first published on Sccopertino
    
    What kind of website is Scoopertino?
    Scoopertino is an imaginary news organization devoted to ferreting out the most relevant stories in the world of Apple..."
    
    Note: The evidence list must be based strictly on the provided inputs. Do not make up information.
    
    Now, generate the QA-format evidence list for the following inputs:
    
    [CLAIM]: {claim}
    [VERDICT]: {verdict}
    [JUSTIFICATION]: {justification}
    [ORIGINAL EVIDENCES LIST]:
    {evidence_context}
    
    Output in QA format:"""
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Chỉ retry nếu là lỗi 429 rate limit
            if "429" not in error_str:
                print(f"Error generating QA (non-retryable): {e}")
                return "What is the evidence for this claim?\nData could not be processed."
            
            # Parse retry_after từ error response
            retry_after = _parse_retry_after(e)
            if retry_after is None:
                # fallback: exponential backoff
                retry_after = min(2 ** attempt, 120)
            else:
                # Thêm 1-2 giây buffer để chắc chắn
                retry_after += 2
            
            print(f"⏳ Rate limited (attempt {attempt}/{max_retries}). Waiting {retry_after:.0f}s...")
            await asyncio.sleep(retry_after)
    
    # Hết số lần retry
    print(f"❌ Failed after {max_retries} retries. Last error: {last_error}")
    return "What is the evidence for this claim?\nData could not be processed."

def build_reverse_evidence_lookup(evidence_vault: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Tạo một từ điển tra cứu ngược: Map từ URL hoặc Text sang mã Doc_id.
    Vì Markdown bị mất mã Doc_id gốc, ta dùng URL/Text để tìm lại mã đó trong vault.
    """
    lookup = {}
    if not isinstance(evidence_vault, list):
        return lookup
        
    for item in evidence_vault:
        citation_map = item.get("citation_map", {})
        for doc_id, doc_data in citation_map.items():
            sentence = doc_data.get("sentence", "")
            url = doc_data.get("url", "")
            
            if isinstance(sentence, list):
                sentence = " ".join([str(s) for s in sentence])
                
            # Lưu cả url và sentence làm key trỏ về doc_id để dễ match
            if url:
                lookup[str(url).strip()] = str(doc_id).strip()
            if sentence:
                lookup[str(sentence).strip()] = str(doc_id).strip()
                
    return lookup

async def process_single_row(line: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        data = line
        
        c_id = data.get("idx", "")
        claim = data.get("claim", "").split("\nDate:")[0].strip()
        label = data.get("final_verdict", "Not Enough Evidence")
        justification = data.get("final_justification", "")
        
        messages = data.get("messages", [])
        judge_message = messages[-1].get("content", "") if messages else ""
        evidence_vault = data.get("evidence_vault", [])
        
        formatted_evidences = []
        
        if "**References:**" in judge_message:
            references_block = judge_message.split("**References:**")[-1]
            
            # Cập nhật regex để bắt cả số thứ tự ref (ví dụ [1]) 
            # Dạng: - **[1]** [Sentence | Domain](URL)
            pattern = r'- \*\*\[([^\]]+)\]\*\* \[(.+?) \| [^\]]+\]\((.+?)\)'
            matches = re.findall(pattern, references_block)
            
            # Tạo bộ tra cứu ngược
            vault_lookup = build_reverse_evidence_lookup(evidence_vault)
            
            for ref_id, sentence, url in matches:
                sentence_clean = sentence.strip()
                url_clean = url.strip()
                
                # Cố gắng tìm mã [Doc_R0...] từ URL hoặc Text. Nếu không thấy, dùng tạm Ref_ID
                real_code = vault_lookup.get(url_clean) or vault_lookup.get(sentence_clean) or f"Ref_{ref_id}"
                
                formatted_evidences.append(f"<{real_code}>: {sentence_clean}")

        # Gửi dữ liệu qua LLM
        if formatted_evidences:
            qa_output = await generate_question(claim, label, justification, formatted_evidences)
            evi_str = qa_output
        else:
            evi_str = ""
            
        return {
            "id": c_id,
            "claim": claim,
            "evi": evi_str,
            "label": label
        }

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy file: {INPUT_FILE}")
        return

    lines = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = json.load(f)

    print(f"Bắt đầu chuyển đổi {len(lines)} claims sang format nộp bài...")
    
    semaphore = asyncio.Semaphore(1) # Xử lý 10 request đồng thời
    tasks = [process_single_row(line, semaphore) for line in lines]
    
    processed_rows = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        processed_rows.append(result)
        
        # Lưu checkpoint sau mỗi 10 claim để không mất dữ liệu nếu script bị gián đoạn
        if len(processed_rows) % 10 == 0:
            _write_csv(processed_rows, is_checkpoint=True)

    print(f"Đang ghi kết quả cuối cùng ra file CSV: {OUTPUT_CSV}")
    _write_csv(processed_rows)

    print("✅ Hoàn tất Convert!")

def _write_csv(rows: List[Dict], is_checkpoint: bool = False):
    """Ghi dữ liệu ra CSV, hỗ trợ checkpoint để tránh mất dữ liệu."""
    output_path = OUTPUT_CSV.replace('.csv', '_checkpoint.csv') if is_checkpoint else OUTPUT_CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['id', 'claim', 'evi', 'label']
        csvfile.write("id,claim,evi,label\n")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        
        rows_sorted = sorted(rows, key=lambda x: int(x['id']))
        for row in rows_sorted:
            row_copy = dict(row)
            row_copy['id'] = int(row_copy['id'])
            writer.writerow(row_copy)

if __name__ == "__main__":
    asyncio.run(main())
