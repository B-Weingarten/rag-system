from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.agent.tools import search_knowledge_base
from src.api.schemas import RouteDecision
from src.config import OLLAMA_HOST, OLLAMA_MODEL

_ROUTER_PROMPT = (
    'Use "rag" only when the question asks about topics covered in this knowledge base: '
    "vector embeddings, similarity metrics, RAG pipelines, chunking, or indexing. "
    'Use "direct" for everything else, including general ML concepts the model already knows.\n'
    "Examples:\n"
    '  "What is cosine similarity?"     -> {{"route": "rag"}}\n'
    '  "What are embeddings?"           -> {{"route": "rag"}}\n'
    '  "How does RAG retrieval work?"   -> {{"route": "rag"}}\n'
    '  "What is 2+2?"                   -> {{"route": "direct"}}\n'
    '  "What is the capital of France?" -> {{"route": "direct"}}\n\n'
    'Return ONLY a JSON object with one key "route" set to "rag" or "direct".\n\n'
    "Question: {query}"
)


class AgentState(TypedDict):
    user_message: str
    route: str            # "rag" | "direct"
    retrieved_chunks: list
    answer: str
    used_rag: bool


def _llm() -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)


def router_node(state: AgentState) -> AgentState:
    prompt = _ROUTER_PROMPT.format(query=state["user_message"])
    llm = ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, format="json")
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        route = RouteDecision(**json.loads(response.content)).route
    except Exception:
        route = "direct"
    return {**state, "route": route}


def rag_node(state: AgentState) -> AgentState:
    chunks = search_knowledge_base(state["user_message"])
    context = "\n\n".join(c["text"] for c in chunks)
    messages = [
        SystemMessage(content="Use only the context below to answer the question. Be concise."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['user_message']}\nAnswer:"),
    ]
    response = _llm().invoke(messages)
    return {
        **state,
        "answer": response.content.strip(),
        "used_rag": True,
        "retrieved_chunks": chunks,
    }


def direct_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="Answer the question directly and briefly."),
        HumanMessage(content=f"Question: {state['user_message']}\nAnswer:"),
    ]
    response = _llm().invoke(messages)
    return {
        **state,
        "answer": response.content.strip(),
        "used_rag": False,
        "retrieved_chunks": [],
    }


_graph = StateGraph(AgentState)
_graph.add_node("router", router_node)
_graph.add_node("rag", rag_node)
_graph.add_node("direct", direct_node)
_graph.set_entry_point("router")
_graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {"rag": "rag", "direct": "direct"},
)
_graph.add_edge("rag", END)
_graph.add_edge("direct", END)
_app = _graph.compile()


def run_agent(user_message: str) -> dict:
    result = _app.invoke(
        {
            "user_message": user_message,
            "route": "",
            "retrieved_chunks": [],
            "answer": "",
            "used_rag": False,
        }
    )
    return {
        "answer": result["answer"],
        "used_rag": result["used_rag"],
        "retrieved_chunks": result["retrieved_chunks"],
    }


async def astream_agent(user_message: str) -> AsyncGenerator[dict, None]:
    initial_state: AgentState = {
        "user_message": user_message,
        "route": "",
        "retrieved_chunks": [],
        "answer": "",
        "used_rag": False,
    }
    state = await asyncio.to_thread(router_node, initial_state)

    if state["route"] == "rag":
        chunks = await asyncio.to_thread(search_knowledge_base, user_message)
        context = "\n\n".join(c["text"] for c in chunks)
        messages = [
            SystemMessage(content="Use only the context below to answer the question. Be concise."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {user_message}\nAnswer:"),
        ]
        used_rag = True
    else:
        chunks = []
        messages = [
            SystemMessage(content="Answer the question directly and briefly."),
            HumanMessage(content=f"Question: {user_message}\nAnswer:"),
        ]
        used_rag = False

    async for chunk in _llm().astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    yield {"type": "done", "used_rag": used_rag, "route": state["route"], "retrieved_chunks": chunks}
