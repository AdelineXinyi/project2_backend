import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def viz_agent(state: AgentState) -> AgentState:
    raw_data = state.get("raw_data", "")
    analysis = state.get("analysis", "")

    user_question = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_question = m.content
            break

    system = SystemMessage(content=(
        "You are a data visualization agent. You receive data and an analysis summary. "
        "Your job: produce a Vega-Lite v5 specification as a JSON object. "
        "Rules:\n"
        "- Always include: $schema, description, data.values, mark, encoding\n"
        "- Add tooltip for interactivity\n"
        "- Use appropriate mark types: bar, line, point, arc\n"
        "- For bar charts, always add a point selection param named 'highlight' "
        "so users can click bars to highlight them. Use condition encoding for color:\n"
        "encoding.color = {condition: {param: 'highlight', value: '#00274c'}, value: '#ccc'}\n"
        "Respond ONLY with a JSON object with two keys:\n"
        "'spec': the Vega-Lite v5 object\n"
        "'summary': one sentence describing what the chart shows\n"
        "No markdown, no explanation, raw JSON only."
    ))

    import json
    try:
        records = json.loads(raw_data)
        if isinstance(records, list):
            truncated = json.dumps(records[:50], ensure_ascii=False)
        else:
            truncated = json.dumps(records, ensure_ascii=False)[:3000]
    except Exception:
        truncated = raw_data[:3000]

    prompt = HumanMessage(content=(
        f"User question: {user_question}\n\n"
        f"Analysis: {analysis}\n\n"
        f"Data (JSON):\n{truncated}\n\n"
        "Return the JSON object with 'spec' and 'summary' keys."
    ))

    response = llm.invoke([system, prompt])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        spec = parsed.get("spec", {})
        summary = parsed.get("summary", analysis)
    except Exception:
        spec = {}
        summary = analysis or "Here is the data."

    if not spec or "data" not in spec or "$schema" not in spec:
        spec = {}

    return {
        "messages": [response],
        "vega_spec": spec,
        "summary": summary,
        "raw_data": raw_data,
        "analysis": analysis,
    }