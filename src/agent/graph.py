from __future__ import annotations

import asyncio
from typing import AsyncGenerator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.agent.tools import search_knowledge_base
from src.config import OLLAMA_HOST, OLLAMA_MODEL


class AgentState(TypedDict):
    user_message: str
    route: str            # "rag" | "direct"
    retrieved_chunks: list
    answer: str
    used_rag: bool


def _llm() -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)


def router_node(state: AgentState) -> AgentState:
    prompt = (
        "Does answering this question require factual knowledge about ML, math, or AI? "
        "Reply with YES or NO only.\n"
        f"Question: {state['user_message']}"
    )
    response = _llm().invoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip().upper()
    route = "rag" if "YES" in answer_text else "direct"
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

    llm = ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    yield {"type": "done", "used_rag": used_rag, "retrieved_chunks": chunks}
