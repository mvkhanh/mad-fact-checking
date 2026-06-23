import asyncio
import json
import os
import argparse
from langgraph_sdk import get_client
from tqdm.asyncio import tqdm

INPUT_FILE = "/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/formatted_inputs/dev.json"
PRE_PRED_FILE = "/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/false_results.json"
OUTPUT_FILE = "/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/dev_results_rerun3.jsonl"
SERVER_URL = "http://localhost:2024"
ASSISTANT_ID = "fact_check_agent"
CONCURRENT_LIMIT = 1

ASSISTANT_ID = "fact_check_agent"
async def process_single_claim(item: dict, client, semaphore: asyncio.Semaphore, pbar: tqdm):
    async with semaphore:
        input_text = item.get("input", "")

        result_data = {
            "input": input_text,
            "error": "Timeout or Execution Failed",
            "thread_id": None
        }

        try:
            thread = await client.threads.create()
            thread_id = thread["thread_id"]
            result_data["thread_id"] = thread_id

            inputs = {
                "messages": [{"role": "user", "content": input_text}],
            }

            await client.runs.wait(
                thread_id=thread_id,
                assistant_id=ASSISTANT_ID,
                input=inputs
            )

            state = await client.threads.get_state(thread_id)
            final_values = state["values"]
            
            raw_messages = final_values.get("messages", [])
            clean_messages = []
            for msg in raw_messages:
                if isinstance(msg, dict):
                    clean_messages.append({
                        "type": msg.get("type", "unknown"),
                        "content": msg.get("content", "")
                    })
                else:
                    clean_messages.append({
                        "type": getattr(msg, "type", "unknown"),
                        "content": getattr(msg, "content", "")
                    })

            result_data = {
                "idx": final_values.get("idx", "-1"),
                "claim": final_values.get("claim", input_text),
                "round_number": final_values.get("round_number", 0),
                "history_summary": final_values.get("history_summary", ""),
                "evidence_vault": final_values.get("evidence_vault", []),
                "final_verdict": final_values.get("final_verdict", "Unknown"),
                "final_justification": final_values.get("final_justification", ""),
                "latest_judge_feedback": final_values.get("latest_judge_feedback", ""),
                "messages": clean_messages,
                "thread_id": thread_id,
                "error": None
            }

        except Exception as e:
            result_data["error"] = str(e)
            print(f"\n❌ Lỗi ở claim ID {input_text[:10]}...: {str(e)}")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            
        pbar.update(1)

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    idx_to_run = set()
    with open(PRE_PRED_FILE, 'r', encoding='utf-8') as f:
        pre_pred = json.load(f)
        for line in pre_pred:
            idx_to_run.add(line["idx"])
    try:
        idx_to_run.remove("78")
    except Exception:
        pass
    data = [d for d in data if d["input"].split('.')[0] in idx_to_run]

    # Khởi tạo Client kết nối với LangGraph Server
    client = get_client(url=SERVER_URL)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    with tqdm(total=len(data), desc="Evaluating via API Server") as pbar:
        tasks = [process_single_claim(item, client, semaphore, pbar) for item in data]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())