import asyncio
import json
import os
import argparse
from langgraph_sdk import get_client
from tqdm.asyncio import tqdm

INPUT_FILE = "/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/formatted_inputs/test.json"
OUTPUT_FILE = "/Users/mvk/FinalProject/FEVER-8-Shared-Task/data_store/averitec/results/test_results.jsonl"
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
    parser = argparse.ArgumentParser(description="Run Evaluation Client with Range")
    parser.add_argument("--start", type=int, default=0, help="Vị trí index bắt đầu (Mặc định: 0)")
    parser.add_argument("--end", type=int, default=None, help="Vị trí index kết thúc (Mặc định: Chạy đến hết)")
    args = parser.parse_args()
    start_idx = args.start
    end_idx = args.end
    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_data_len = len(data)
    actual_end = end_idx if end_idx is not None else total_data_len
    if start_idx < 0 or start_idx >= total_data_len:
        print(f"❌ Lỗi: start_idx ({start_idx}) vượt quá giới hạn dữ liệu (0 -> {total_data_len-1})")
        return
        
    data_to_run = data[start_idx:actual_end]
    print(f"📌 Đã tải {total_data_len} claims. Đang lấy vùng dữ liệu từ index {start_idx} đến {actual_end} (Tổng: {len(data_to_run)} claims)")
    if not data_to_run:
        print("Không có dữ liệu nào trong khoảng index này.")
        return
    # Resume logic: Đọc những câu đã chạy để bỏ qua
    processed_inputs = set()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_input = json.loads(line)
                        processed_inputs.add(f'{processed_input["idx"]}. {processed_input["claim"]}')
                    except Exception as e:
                        pass
    
    remaining_data = [item for item in data_to_run if item.get("input") not in processed_inputs]
    print(f"Tổng: {len(data_to_run)} | Đã chạy: {len(processed_inputs)} | Cần chạy: {len(remaining_data)}")

    if not remaining_data:
        return
    for r in remaining_data:
        print(r['input'].split('.')[0])
    # Khởi tạo Client kết nối với LangGraph Server
    client = get_client(url=SERVER_URL)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    with tqdm(total=len(remaining_data), desc="Evaluating via API Server") as pbar:
        tasks = [process_single_claim(item, client, semaphore, pbar) for item in remaining_data]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())