from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def analysis_agent(state: AgentState) -> AgentState:
    raw_data = state.get("raw_data", "")
    if not raw_data or raw_data.strip() in ("", "[]", "{}"):
        return {
            "messages": state["messages"],
            "analysis": "no data retrieved",
            "raw_data": raw_data,
            "vega_spec": {},
            "summary": "no data retrieved",
        }

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

    import json
    try:
        records = json.loads(raw_data)
        if isinstance(records, list):
            truncated = json.dumps(records[:60], ensure_ascii=False)
        else:
            truncated = json.dumps(records, ensure_ascii=False)[:4000]
    except Exception:
        truncated = raw_data[:4000]

    prompt = HumanMessage(content=(
        f"User question: {user_question}\n\n"
        f"Raw data:\n{truncated}\n\n"
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