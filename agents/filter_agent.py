from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from .tools import ALL_TOOLS
from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def filter_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content=(
        "You are a data filtering agent for University of Michigan research papers. "
        "You have access to tools that query a dataset of UMich papers. "
        "Your ONLY job: read the user's question and call exactly ONE tool "
        "that retrieves the most relevant data. Do not analyze or explain — just call the tool."
    ))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {
        "messages": [response],
        "raw_data": "",
        "analysis": "",
        "vega_spec": {},
        "summary": "",
    }


def should_call_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "call_tool"
    return "analysis_agent"


def call_tool(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]
    tool_fn = TOOL_MAP[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )

    return {
        "messages": [tool_msg],
        "raw_data": str(result),
        "analysis": "",
        "vega_spec": {},
        "summary": "",
    }