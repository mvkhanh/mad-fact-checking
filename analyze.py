import os
import json
import asyncio
from typing import Literal
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('/Users/mvk/FinalProject/FEVER-8-Shared-Task/app/backend/.env')

# Cấu hình
INPUT_FILE = '/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/incorrect_predictions_rerun.json'
OUTPUT_FILE = '/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/error_analysis_rerun.json'
LABEL_FILE = '/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/dev.json'
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    max_tokens=512,
    max_retries=2,
    temperature=0.0
)

META_JUDGE_PROMPT = """
You are an expert AI Evaluator specializing in Fact-Checking Pipelines. 
Your task is to analyze a failed prediction made by a Fact-Checking Agent and identify the ROOT CAUSE of the failure.
You will be provided the claim; Fact-Checking Agent's retrieved evidence, verdict, justification and ground truth verdict, justification, evidence in Question Answer (QA) format.

Analyze why the Agent failed to predict the Ground Truth Verdict. Categorize the root cause into one of the following:
1. Retrieval Failure: Did the system fail to find relevant evidence?
2. Reasoning/Logic Error: Did the agent misinterpret the text, fail at math/dates, or jump to conclusions?
3. Multi-part Falsification Failure: Did the agent ignore a false sub-claim just because the main claim was true?
4. External Knowledge Bias: Did the agent hallucinate or rely on outside knowledge instead of the provided evidence?
5. Definition Mismatch: Is the agent using a different definition of "Supported/Refuted/Not Enough Evidence/Conflicting Evidence/Cherrypicking" than the Ground Truth?
6. Other: Other cause of error.

Output strictly in JSON matching the provided schema.
"""

class OutputFormat(BaseModel):
    root_cause_category: Literal[
        "Retrieval Failure", 
        "Reasoning/Logic Error", 
        "Multi-part Falsification Failure", 
        "External Knowledge Bias", 
        "Definition Mismatch",
        "Other"
    ] = Field(
        description="The exact name of the category from the list above."
    )
    
    explanation: str = Field(
        description="A 1-2 sentence explanation of exactly where the agent's logic broke down."
    )

llm = llm.with_structured_output(OutputFormat)

async def analyze_error(item, ground_truth_item, semaphore):
    async with semaphore:
        if item['idx'] == '78':
            print('Skip 78')
            return item
        evidence_summary = ""
        for ev in item.get("evidence_vault", []):
            evidence_summary += f"[{ev.get('stance')}]: "
            for key, val in ev.get("citation_map", {}).items():
                evidence_summary += f"- {val.get('sentence', '')}\n"
        ground_truth_questions = ''
        for i, qa in enumerate(ground_truth_item['questions'], 1):
            ground_truth_questions += f'Question {i}: {qa["question"]}\n'
            for j, ans in enumerate(qa['answers'], 1):
                ground_truth_questions += f'Answer {i}.{j}: {ans["answer"]}\n'
            ground_truth_questions += '\n'
        # Điền data vào Prompt
        prompt = f"""
        [CLAIM]: {item.get('claim')}
        [AGENT'S RETRIEVED EVIDENCE]: {evidence_summary}
        [AGENT'S VERDICT]: {item.get('final_verdict')}
        [AGENT'S JUSTIFICATION]: {item.get('final_justification')}
        [GROUND TRUTH VERDICT]: {ground_truth_item.get('label')}
        [GROUND TRUTH JUSTIFICATION]: {ground_truth_item.get('justification')}
        [GROUND TRUTH EVIDENCE IN QA FORMAT]: {ground_truth_questions}
        """

        try:
            response = await llm.ainvoke([
                    {"role": "system", "content": META_JUDGE_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                
            )
            
            item["error_analysis"] = response.model_dump()
            return item
            
        except Exception as e:
            print(f"Error analyzing idx {item.get('idx')}: {e}")
            item["error_analysis"] = {"root_cause_category": "API Error", "explanation": str(e)}
            return item

async def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    print(f"Bắt đầu phân tích {len(data)} lỗi sai...")
    semaphore = asyncio.Semaphore(1)
    
    tasks = [analyze_error(item, label_data[int(item['idx'])], semaphore) for item in data]
    
    analyzed_data = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        analyzed_data.append(result)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analyzed_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Hoàn tất! Đã lưu báo cáo phân tích tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())