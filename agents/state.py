from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_data: str
    analysis: str
    vega_spec: dict
    summary: str