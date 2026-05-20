from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def analysis_agent(state: AgentState) -> AgentState:
    raw_data = state.get("raw_data", "")

    user_question = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_question = m.content
            break

    system = SystemMessage(content=(
        "You are a data analysis agent. You receive raw JSON data from a query "
        "about University of Michigan research papers. "
        "Your job: interpret the numbers, identify the key trend or finding, "
        "and write a clear 2-3 sentence insight summary. "
        "Be specific — mention actual numbers. Do not generate any visualization."
    ))

    prompt = HumanMessage(content=(
        f"User question: {user_question}\n\n"
        f"Raw data:\n{raw_data[:4000]}\n\n"
        "Write a 2-3 sentence analysis of what this data shows."
    ))

    response = llm.invoke([system, prompt])

    return {
        "messages": [response],
        "analysis": response.content.strip(),
        "raw_data": raw_data,
        "vega_spec": {},
        "summary": "",
    }