from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .state import AgentState
from .filter_agent import filter_agent, should_call_tool, call_tool
from .analysis_agent import analysis_agent
from .viz_agent import viz_agent


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("filter_agent", filter_agent)
    g.add_node("call_tool", call_tool)
    g.add_node("analysis_agent", analysis_agent)
    g.add_node("viz_agent", viz_agent)

    g.set_entry_point("filter_agent")

    g.add_conditional_edges("filter_agent", should_call_tool, {
        "call_tool": "call_tool",
        "analysis_agent": "analysis_agent",
    })

    g.add_edge("call_tool", "analysis_agent")
    g.add_edge("analysis_agent", "viz_agent")
    g.add_edge("viz_agent", END)

    return g.compile()


graph = build_graph()


def run_query(user_question: str) -> dict:
    result = graph.invoke({
        "messages": [HumanMessage(content=user_question)],
        "raw_data": "",
        "analysis": "",
        "vega_spec": {},
        "summary": "",
    })

    return {
        "summary": result["summary"],
        "vega_spec": result["vega_spec"],
    }