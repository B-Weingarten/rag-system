from typing import Literal

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str


_RAG_TOKENS = {"rag", "ml", "ai", "ml/ai", "mlai", "yes", "true", "1", "specialized", "technical"}


class RouteDecision(BaseModel):
    route: Literal["rag", "direct"]

    @field_validator("route", mode="before")
    @classmethod
    def coerce_route(cls, v: object) -> str:
        s = str(v).strip().lower() if v is not None else ""
        return "rag" if s in _RAG_TOKENS else "direct"
