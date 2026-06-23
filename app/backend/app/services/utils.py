import re
import json
import html as _html
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple
async def translate_to_vi(text: str) -> str:
    if not text:
        return text
    try:
        from app.services.llm import llm_service
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm_service.call(messages=[
            SystemMessage(content=(
                "Translate the following text to Vietnamese. "
                "Preserve all Markdown formatting, URLs, and citation tags like [1] or [Doc_X] exactly as-is. "
                "Output only the translated text.\n\n"
                "Use these fixed translations for domain-specific terms:\n"
                "- Debate → Tranh luận\n"
                "- Proponent → Tác nhân Ủng hộ\n"
                "- Opponent → Tác nhân Bác bỏ\n"
                "- Judge → Tác nhân Thẩm phán\n"
                "- Verdict → Nhãn phán quyết\n"
                "- Justification → Giải trình"
            )),
            HumanMessage(content=text)
        ])
        return response.content
    except Exception as e:
        print(f"[translate_to_vi] Warning: {e}")
        return text

def escape_ui_text(text: str) -> str:
    if not text:
        return text
    return text.replace('$', '&#36;')

def get_domain(url: str) -> str:
    """
    Safely extracts domain from URL. 
    Handles Web Archive URLs by extracting the original target domain.
    """
    if not url or url == "#": 
        return "Unknown Source"
    
    try:
        if "web.archive.org/web/" in url:
            match = re.search(r'web\.archive\.org/web/\d+/(https?://.+)', url)
            if match:
                original_url = match.group(1)
                domain = urlparse(original_url).netloc
            else:
                domain = urlparse(url).netloc
        else:
            domain = urlparse(url).netloc

        if domain.startswith("www."):
            domain = domain[4:]
            
        return domain or "Unknown Source"
        
    except Exception:
        return "Unknown Source"

def extract_text_from_content(content: Any) -> str:
    """Extracts raw text from LangChain message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
    return str(content)

def format_arg_for_prompt(arg_dict: Dict[str, Any]) -> str:
    """Formats reasoning dictionary into a clean text block for LLM prompts."""
    if not arg_dict:
        return "None provided."
    verdict = arg_dict.get("verdict", "N/A")
    justification = arg_dict.get("justification", "N/A")
    return f"- Verdict: {verdict}\n- Justification: {justification}"

def build_evidence_dossier(arg_dict: dict, agent_name: str, master_dict: dict) -> str:
    """Builds the evidence text block for the Judge based on agent citations."""
    dossier = f"\n{agent_name.upper()} cited evidence\n"
    cited_tags = arg_dict.get("cited_evidence", [])
    
    if not cited_tags:
        return dossier + "(No specific evidence explicitly cited by agent.)\n"
        
    has_valid_tag = False
    for tag in cited_tags:
        formatted_tag = f"[{tag}]" if not tag.startswith("[") else tag
        if formatted_tag in master_dict:
            val = master_dict[formatted_tag]
            sentence = val.get("sentence", "N/A") if isinstance(val, dict) else "N/A"
            url = val.get("url", "#") if isinstance(val, dict) else val
            domain = get_domain(url)
            dossier += f"- {formatted_tag} (Source: {domain}): {sentence}\n"
            has_valid_tag = True
            
    if not has_valid_tag:
         dossier += "(Agent provided invalid tags or hallucinated citations.)\n"
    return dossier

def process_retrieved_evidence(retrieved_content: str, current_round: int, stance: str, queries: List[str], lang: str = "en") -> Tuple[str, str, dict, dict]:
    """Processes search results into Prompt format and UI Markdown."""
    stance_prefix = "Prop" if stance == "Proponent" else "Opp"
    formatted_evidence_for_prompt = ""
    citation_map = {}
    tag_to_num_ui = {}

    lbl_queries = "- **Truy Vấn:**\n" if lang == "vi" else "- **Queries:**\n"
    lbl_none    = "(Không tìm thấy bằng chứng)\n" if lang == "vi" else "(No evidence found)\n"
    lbl_err     = "*(Lỗi phân tích bằng chứng)*\n" if lang == "vi" else "*(Error parsing evidence)*\n"

    md_block = lbl_queries + "".join([f"    - {q}\n" for q in queries]) + "\n"

    try:
        retrieved_list = json.loads(retrieved_content)
        if not retrieved_list:
            md_block += lbl_none
        else:
            count_lbl = (
                f"**{len(retrieved_list)} bằng chứng tìm được:**\n\n"
                if lang == "vi" else
                f"**{len(retrieved_list)} evidence results:**\n\n"
            )
            md_block += count_lbl
            for i, item in enumerate(retrieved_list, start=1):
                raw_doc_id = f"Doc_R{current_round}_{stance_prefix}_{i}"
                doc_id = f"[{raw_doc_id}]"
                sentence = item.get("sentence", "").replace("\n", " ")
                url = item.get("url", "#")
                domain = get_domain(url)

                citation_map[doc_id] = {"url": url, "sentence": sentence}
                formatted_evidence_for_prompt += f"  {doc_id} (Source: {domain}): {sentence}\n"
                tag_to_num_ui[raw_doc_id] = str(i)

                # First sentence as collapsed preview
                parts = sentence.split(". ")
                preview = parts[0]
                if len(preview) > 130:
                    preview = preview[:130].rsplit(" ", 1)[0] + "..."
                elif len(parts) > 1:
                    preview += "..."

                md_block += (
                    f"<details>\n"
                    f"<summary><strong>[{i}]</strong> "
                    f'<a href="{_html.escape(url, quote=True)}">{_html.escape(domain)}</a>'
                    f" — {_html.escape(preview)}</summary>\n"
                    f"<p>{_html.escape(sentence)}</p>\n"
                    f"</details>\n\n"
                )
    except json.JSONDecodeError:
        md_block += lbl_err

    return formatted_evidence_for_prompt, md_block, citation_map, tag_to_num_ui

def clean_single(m): 
    return "[" + ", ".join(list(dict.fromkeys(re.findall(r'\d+', m.group(0))))) + "]"

def clean_multi(m):
    nums = list(dict.fromkeys(re.findall(r'\[(\d+)\]', m.group(0))))
    return ", ".join([f"[{n}]" for n in nums]) if "," in m.group(0) else "".join([f"[{n}]" for n in nums])

def build_resolve_ui(final_verdict: str, justification: str, master_citation_map: dict, lang: str = "en") -> Tuple[str, str]:
    """Formats the Judge's final resolution for UI."""
    found_tags_raw = re.findall(r'Doc_R\d+_(?:Prop|Opp)_\d+', justification)
    raw_tags_unique = sorted(list(set(found_tags_raw)))
    
    formatted_justification = justification
    if not raw_tags_unique and master_citation_map:
        raw_tags_unique = sorted([key.strip("[]") for key in master_citation_map.keys()])
        note = "\n\n*(Lưu ý: Thẩm phán không trích dẫn. Tài liệu tham khảo được liệt kê bên dưới).*" if lang == "vi" else "\n\n*(Note: Inline citations omitted by Judge. Consulted sources listed below).*"
        formatted_justification += note

    content_to_num = {}
    tag_to_num_raw = {}
    unique_refs = {}
    ui_counter = 1
    for raw_tag in raw_tags_unique:
        map_key = f"[{raw_tag}]"
        if map_key in master_citation_map:
            val = master_citation_map[map_key]
            url = val.get("url", "#") if isinstance(val, dict) else val
            sentence = val.get("sentence", "Link").replace('\n', ' ').strip() if isinstance(val, dict) else "Link"
            domain = get_domain(url)
            content_sig = f"{sentence}:::{url}"
            
            if content_sig not in content_to_num:
                num_str = str(ui_counter)
                content_to_num[content_sig] = num_str
                unique_refs[num_str] = {"sentence": sentence, "url": url, "domain": domain}
                ui_counter += 1
                
            tag_to_num_raw[raw_tag] = content_to_num[content_sig]

    for old_tag in sorted(tag_to_num_raw.keys(), key=len, reverse=True):
        formatted_justification = formatted_justification.replace(old_tag, tag_to_num_raw[old_tag])

    formatted_justification = re.sub(r'\[\d+(?:[\s,]+\d+)+\]', clean_single, formatted_justification)
    formatted_justification = re.sub(r'\[\d+\](?:[\s,]*\[\d+\])+', clean_multi, formatted_justification)
    if lang == "vi":
        ui_text = f"### Kết Luận: `{final_verdict}`\n\n**Giải Trình Của Thẩm Phán:**\n{formatted_justification}\n\n**Tài Liệu Tham Khảo:**\n"
    else:
        ui_text = f"### Final Verdict: `{final_verdict}`\n\n**Judge's Justification:**\n{formatted_justification}\n\n**References:**\n"
    has_refs = False

    for num_str, ref in unique_refs.items():
        ui_text += f"- **[{num_str}]** [{ref['sentence']} | {ref['domain']}]({ref['url']})\n"
        has_refs = True
            
    if not has_refs:
        ui_text += ("- Không có tài liệu nào được trích dẫn.\n" if lang == "vi" else "- No external URLs cited.\n")
        
    return escape_ui_text(formatted_justification), ui_text