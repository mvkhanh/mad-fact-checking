"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""
import re
import json
import asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command
from langgraph.types import RunnableConfig
from app.core.langgraph.prompt import *

def _t(web_mode: bool, en: str, vi: str) -> str:
    """Return vi string in web/demo mode, en string in KB mode."""
    return vi if web_mode else en
from app.core.langgraph.tools import tools
from app.schemas import GraphState, StandardQueryOutput, HyDEQueryOutput, AgentReasoningOutput, JudgeOutput
from app.services import *
from app.core.config import settings

tools_by_name = {tool.name: tool for tool in tools}

async def init_debate(state: GraphState, config: RunnableConfig) -> Command:
    """Extracts claim and ID from the initial message without calling LLM."""
    raw_text = extract_text_from_content(state.messages[0].content).strip()
    match = re.match(r'^(\d+)[\.\-\:]\s*(.*)', raw_text, re.DOTALL)
    if match:
        claim_idx = str(match.group(1))
        clean_claim = match.group(2).strip()
    else:
        # No index → demo mode: use web search instead of knowledge store
        claim_idx = ""
        clean_claim = raw_text

    web_mode = not claim_idx
    header = _t(web_mode, "## Initialized Debate for Claim\n", "## Bắt Đầu Kiểm Tra Thông Tin\n")
    return Command(
        update={
            "claim": clean_claim,
            "idx": claim_idx,
            "messages": [AIMessage(header + clean_claim)]
        },
        goto=["proponent_agent", "opponent_agent"]
    )
    
async def _base_debater_agent(state: GraphState, stance: str, query_gen_prompt: str, reasoning_prompt: str) -> dict:
    """Base function for debaters."""
    web_mode = not state.idx
    lang = "vi" if web_mode else "en"
    current_round = state.round_number
    opp_name = "Opponent" if stance == "Proponent" else "Proponent"
    opp_latest_arg = state.opponent_latest_arg if stance == "Proponent" else state.proponent_latest_arg
    opp_arg_str = json.dumps(opp_latest_arg, ensure_ascii=False, indent=2) if opp_latest_arg else "None"

    # 1. Extract context
    previous_queries, excluded_sentences = [], []
    
    if current_round > 0 and state.evidence_vault:
        for ev in state.evidence_vault:
            if ev.get("stance") == stance and ev.get("queries_used"):
                previous_queries.append(ev["queries_used"])
            citation_map = ev.get("citation_map", {})
            for val in citation_map.values():
                excluded_sentences.append(val.get("sentence"))

    # 2. Generate queries
    human_query_content = f"Original Claim: {state.claim}\n"
    if current_round > 0:
        human_query_content += f"\n\n--- ROUND {current_round} CONTEXT ---"
        if state.history_summary:
            human_query_content += f"\nOverall Debate History:\n{state.history_summary}"
        if state.latest_judge_feedback:
            human_query_content += f"\n\n**Judge's last round directives:\n{state.latest_judge_feedback}"
        human_query_content += f"\n\n{opp_name}'s Latest Arguments (Target to counter):\n{opp_arg_str}"
        human_query_content += f"\n\nYour Previous Queries (DO NOT REPEAT):\n{', '.join(previous_queries)}"

    target_schema = HyDEQueryOutput if current_round == 0 else StandardQueryOutput
    sys_query = query_gen_prompt + (VI_LANG_INSTRUCTION if web_mode else "")
    try:
        ai_query_msg = await llm_service.call(
            messages=[SystemMessage(content=sys_query), HumanMessage(content=human_query_content)],
            pydantic_model=target_schema
        )
        query_output = json.loads(ai_query_msg.content)
        queries = query_output.get("hypothetical_excerpts", []) if current_round == 0 else query_output.get("queries", [])
    except Exception as e:
        print(f"Error in query generation: {e}")
        queries = [state.claim[:100]]
    if web_mode:
        queries = queries[:3]
    
    queries_used_str = escape_ui_text(", ".join(queries))

    # 3. Evidence Retrieval
    tool_calls = [{
        "name": "evidence_retrieval_tool", 
        "args": {
            "fact": state.claim,
            "queries": queries, 
            "idx": state.idx,
            "excluded_sentences": excluded_sentences
        },
    }]         
    tool_outputs = await execute_tool_calls_concurrently(tool_calls, tools_by_name)
    prompt_evidence, md_block, citation_map, tag_to_num_ui = process_retrieved_evidence(
        tool_outputs[0].content, current_round, stance, queries, lang=lang
    )
    
    new_evidence = [{"stance": stance, "citation_map": citation_map, "queries_used": queries_used_str}]

    # 4. Reasoning
    human_reasoning_content = f"Original Claim: {state.claim}\n\nRetrieved Evidence:\n{prompt_evidence}\n"    
    if current_round > 0:
        human_reasoning_content += f"\n--- ROUND {current_round} CONTEXT ---\n"
        if state.latest_judge_feedback:
            human_reasoning_content += f"**Judge's Directives:**\n{state.latest_judge_feedback}\n\n"
        human_reasoning_content += f"{opp_name}'s Latest Arguments:\n{opp_arg_str}\n\n"
            
    sys_reasoning = reasoning_prompt + (VI_LANG_INSTRUCTION if web_mode else "")
    try:
        ai_reasoning_msg = await llm_service.call(
            messages=[SystemMessage(content=sys_reasoning), HumanMessage(content=human_reasoning_content)],
            pydantic_model=AgentReasoningOutput
        )
        reasoning_output = json.loads(ai_reasoning_msg.content)
    except Exception as e:
        print(f"Error in {stance} Reasoning: {e}")
        reasoning_output = {"verdict": "Not Enough Info", "justification": "Failed to parse reasoning."}

    raw_justification = reasoning_output.get('justification', 'N/A')
    ui_justification = raw_justification
    for raw_tag, num_tag in tag_to_num_ui.items():
        ui_justification = ui_justification.replace(raw_tag, num_tag)
    ui_justification = escape_ui_text(ui_justification)
    stance_label = _t(web_mode, stance, "Bên Ủng Hộ" if stance == "Proponent" else "Bên Phản Đối")
    round_label = _t(web_mode, f"Round {current_round}", f"Vòng {current_round}")
    reasoning_lbl = _t(web_mode, "### Reasoning Output", "### Kết Quả Phân Tích")
    ui_msg = escape_ui_text(f"## {stance_label} ({round_label})\n\n{md_block}\n{reasoning_lbl}\n- **Verdict:** `{reasoning_output.get('verdict', 'N/A')}`\n- **Justification:** {ui_justification}")
    return {
        f"{stance.lower()}_latest_arg": reasoning_output,
        "evidence_vault": new_evidence,
        "messages": [AIMessage(content=ui_msg.strip())]
    }

async def proponent_agent(state: "GraphState", config: RunnableConfig) -> Command:
    """Node Agent"""
    updates = await _base_debater_agent(state=state, stance="Proponent", query_gen_prompt=PROPONENT_QUERY_GEN_PROMPT if state.round_number > 0 else PROPONENT_HYDE_PROMPT, reasoning_prompt=PROPONENT_REASONING_PROMPT)
    return Command(update=updates, goto="judge_agent")

async def opponent_agent(state: "GraphState", config: RunnableConfig) -> Command:
    """Node Agent"""
    updates = await _base_debater_agent(state=state, stance="Opponent", query_gen_prompt=OPPONENT_QUERY_GEN_PROMPT if state.round_number > 0 else OPPONENT_HYDE_PROMPT, reasoning_prompt=OPPONENT_REASONING_PROMPT)
    return Command(update=updates, goto="judge_agent")

async def judge_agent(state: "GraphState", config: RunnableConfig) -> Command:
    """Judge Agent: Evaluates debaters' arguments and routes facts."""
    web_mode = not state.idx
    lang = "vi" if web_mode else "en"
    current_round = state.round_number
    # master_dict: {"[Doc_R0_Prop_1]": {"sentence": "...", "url": "..."}}
    master_dict = {}
    if state.evidence_vault:
        for ev in state.evidence_vault:
            master_dict.update(ev.get("citation_map", {}))

    human_content = f"--- ROUND {current_round} ---\n\nOriginal Claim: {state.claim}\n\n"
    if state.history_summary:
        human_content += f"Debate History Context:\n{state.history_summary}\n\n"

    human_content += f"Proponent's Evaluation:\n{format_arg_for_prompt(state.proponent_latest_arg)}\n\n"
    human_content += build_evidence_dossier(state.proponent_latest_arg or {}, "Proponent", master_dict)
    human_content += f"Opponent's Evaluation:\n{format_arg_for_prompt(state.opponent_latest_arg)}\n\n"
    human_content += build_evidence_dossier(state.opponent_latest_arg or {}, "Opponent", master_dict)

    sys_judge = JUDGE_PROMPT + (VI_LANG_INSTRUCTION if web_mode else "")
    try:
        ai_msg = await llm_service.call(
            messages=[SystemMessage(content=sys_judge), HumanMessage(content=human_content)],
            pydantic_model=JudgeOutput
        )
        judge_output = json.loads(ai_msg.content)
    except Exception as e:
        print(f"Error in Judge Agent: {e}")
        judge_output = {"action": "RETRY", "final_verdict": "Not Enough Evidence", "judge_justification": "Parse error."}

    action = judge_output.get("action", "RETRY")
    final_verdict = judge_output.get("final_verdict", "Not Enough Evidence")
    justification = judge_output.get("judge_justification", "")
    tasks = judge_output.get("investigative_tasks", [])
    
    if action == "RETRY" and current_round >= settings.MAX_ROUND_DEBATE:
        action = "RESOLVE"
        final_verdict = "Not Enough Evidence"
        justification = f"Reached maximum rounds ({settings.MAX_ROUND_DEBATE}). " + justification

    if action == "RESOLVE":
        formatted_just, ui_format_text = build_resolve_ui(final_verdict, justification, master_dict, lang=lang)
        ui_format_text = escape_ui_text(ui_format_text)
        formatted_just = escape_ui_text(formatted_just)
        return Command(
            update={"final_verdict": final_verdict, "final_justification": formatted_just, "messages": [AIMessage(content=ui_format_text)]},
            goto=END
        )

    else:
        retry_hdr = _t(web_mode, f"### Judge Directs a RETRY (Round {current_round})\n\n**Judge's Analysis:**\n", f"### Thẩm Phán Yêu Cầu Tiếp Tục (Vòng {current_round})\n\n**Phân Tích Của Thẩm Phán:**\n")
        ui_format_text = retry_hdr + justification + "\n\n"
        feedback_for_agents = f"Judge's analysis: {justification}\n"
        if tasks:
            tasks_hdr = _t(web_mode, "**Investigative Tasks for Next Round:**\n", "**Nhiệm Vụ Điều Tra Vòng Tiếp:**\n")
            ui_format_text += tasks_hdr + "".join([f"- {t}\n" for t in tasks])
            feedback_for_agents += "REQUIRED TASKS:\n" + "\n".join([f"- {t}" for t in tasks])
        ui_format_text = escape_ui_text(ui_format_text)

        return Command(
            update={"round_number": current_round + 1, "latest_judge_feedback": feedback_for_agents, "messages": [AIMessage(content=ui_format_text)]},
            goto="history_summarizer"
        )

async def history_summarizer(state: "GraphState", config: RunnableConfig) -> Command:
    """Summarizes the latest debate round to maintain concise context."""
    web_mode = not state.idx
    current_round = state.round_number - 1
    content = f"Summarize Round {current_round}:\n\nProponent's Evaluation:\n{format_arg_for_prompt(state.proponent_latest_arg)}\n\nOpponent's Evaluation:\n{format_arg_for_prompt(state.opponent_latest_arg)}\n\nJudge's Analysis:{state.latest_judge_feedback}"

    sys_hist = HISTORY_SUMMARIZER_PROMPT + (VI_LANG_INSTRUCTION if web_mode else "")
    try:
        ai_msg = await llm_service.call(messages=[SystemMessage(content=sys_hist), HumanMessage(content=content)])
        new_summary = f"--- ROUND {current_round} SUMMARY ---\n{ai_msg.content.strip()}\n\n"
        updated_history = state.history_summary + new_summary
    except Exception as e:
        print(f"Error in History Summarizer: {e}")
        updated_history, new_summary = state.history_summary, "Error summarizing."

    hist_hdr = _t(web_mode, f"## History Summarizer (Round {current_round})\n", f"## Tóm Tắt Vòng {current_round}\n")
    return Command(
        update={"history_summary": updated_history, "messages": [AIMessage(content=hist_hdr + new_summary)]},
        goto=["proponent_agent", "opponent_agent"]
    )

async def execute_tool_calls_concurrently(tool_calls: List[Dict[str, Any]], tools_dict: Dict[str, Any]) -> List[ToolMessage]:
    """
    Executes multiple tool calls concurrently.
    Based on the provided sample logic.
    """
    async def _execute_tool(tool_call: dict) -> ToolMessage:
        tool_instance = tools_dict[tool_call["name"]]
        tool_result = await tool_instance.ainvoke(tool_call["args"])
        return ToolMessage(content=str(tool_result), name=tool_call["name"], tool_call_id=tool_call.get("id", "default_id"),)

    # Execute tool calls concurrently when multiple are requested
    if len(tool_calls) == 1:
        return [await _execute_tool(tool_calls[0])]
    else:
        return list(await asyncio.gather(*[_execute_tool(tc) for tc in tool_calls]))

builder = StateGraph(GraphState)
builder.add_node("init_debate", init_debate)
builder.add_node("proponent_agent", proponent_agent)
builder.add_node("opponent_agent", opponent_agent)
builder.add_node("judge_agent", judge_agent)
builder.add_node("history_summarizer", history_summarizer)
builder.set_entry_point("init_debate")

graph = builder.compile()