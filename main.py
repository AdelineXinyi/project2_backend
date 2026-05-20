"""
main.py
-------
FastAPI backend. Exposes:
  POST /chat          - send a question, get back summary + vega-lite spec
  GET  /health        - sanity check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agents.graph import run_query

app = FastAPI(title="SciSciNet Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    summary:   str
    vega_spec: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_query(req.question)
    return ChatResponse(
        summary=result["summary"],
        vega_spec=result["vega_spec"],
    )