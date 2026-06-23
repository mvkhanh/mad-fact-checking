from pydantic import BaseModel, Field
from typing import List, Optional

class StandardQueryOutput(BaseModel):
    queries: List[str] = Field(
        description="List of 2-8 specific search queries investigating different facets of the claim.",
        min_length=2, max_length=8
    )

class HyDEQueryOutput(BaseModel):
    hypothetical_excerpts: List[str] = Field(
        description="A list of hypothetical excerpts (HyDE). Each excerpt must be maximum 2 sentences and under 40 words.",
        min_length=2, max_length=4
    )

class AgentReasoningOutput(BaseModel):
    verdict: str = Field(
        description="Verdict for the claim.",
        enum=["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    )
    justification: str = Field(
        description="Logical justification base on retrieved evidence, context, ... Must citing [Doc_X] tags in cited_evidence. Maximum 4 sentences."
    )

    cited_evidence: List[str] = Field(
        description="A list of exact [Doc_X] tags (e.g., ['[Doc_R0_Prop_1]', '[Doc_R0_Prop_3]']) that directly support your justification. Empty list if none.",
        default=[]
    )

class JudgeOutput(BaseModel):
    action: str = Field(
        description="Choose 'RESOLVE' to end the debate or 'RETRY' to seek more evidence.",
        enum=["RESOLVE", "RETRY"]
    )
    final_verdict: Optional[str] = Field(
        description="Required if action is RESOLVE. Choose the final status of the claim.",
        enum=["Supported", "Refuted", "Conflicting Evidence/Cherrypicking", "Not Enough Evidence"],
        default=None
    )
    judge_justification: str = Field(
        description="If RESOLVE, this is the final explanation. If RETRY, explain the logical gaps. Both case must citing [Doc_X] tags in provided evidence. Maximum 3 sentences."
    )
    investigative_tasks: Optional[List[str]] = Field(
        description="If action is RETRY, provide specific tasks or questions for the agents to investigate in the next round.",
        default=[]
    )