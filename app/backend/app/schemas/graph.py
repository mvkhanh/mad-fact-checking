import operator
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class GraphState(BaseModel):
    """
    State definition for the Misinformation Detection LangGraph Workflow.
    Manages the overall state of the multi-agent debate, including facts, 
    retrieved evidence, and temporary/final arguments.
    """

    idx: str = Field(
        default="0", 
        description="The unique identifier (ID) extracted from the input claim."
    )

    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, 
        description="The sequence of messages in the conversation."
    )

    claim: str = Field(
        default="", 
        description="The original input claim to be verified."
    )

    round_number: int = Field(
        default=0, 
        description="The current iteration round of the debate."
    )

    history_summary: str = Field(
        default="", 
        description="Summary of the debate history and arguments from previous rounds."
    )

    evidence_vault: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list, 
        description="Storage for retrieved evidence."
    )

    proponent_latest_arg: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="The latest argument from the proponent agent, containing the pseudo-verdict and justification."
    )

    opponent_latest_arg: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="The latest argument from the opponent agent, containing the pseudo-verdict and justification."
    )

    final_verdict: str = Field(
        default="", 
        description="The final status of the claim."
    )

    final_justification: str = Field(
        default="",
        description="The detailed explanation for the final verdict."
    )

    latest_judge_feedback: str = Field(
        default="", 
        description="Judge's feedback on claim for what to process in the next round."
    )