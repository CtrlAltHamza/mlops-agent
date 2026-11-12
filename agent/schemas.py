"""Shared result types for agent tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    tool: str
    data: dict[str, Any] | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    status: str = "ok"


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    retraining_started: bool = False

    def render(self) -> str:
        return format_sections(
            self.answer,
            self.facts,
            self.uncertainties,
            self.recommended_actions,
        )


class ChatRequest(BaseModel):
    message: str
    request_id: str | None = None


def format_sections(
    summary: str,
    facts: list[str],
    uncertainties: list[str],
    actions: list[str],
) -> str:
    def block(title: str, items: list[str]) -> str:
        body = "\n".join(f"- {item}" for item in items) if items else "- None."
        return f"{title}:\n{body}"

    return "\n\n".join(
        [
            summary.strip(),
            block("Facts", facts),
            block("Uncertainties", uncertainties),
            block("Recommended actions", actions),
        ]
    )


class InvalidRequest(Exception):
    """The user request cannot be executed as stated."""


class ToolUnavailable(Exception):
    """An external system did not return evidence."""


class ConfirmationRequired(Exception):
    """A state-changing operation needs an explicit confirmation."""


class NotAuthorized(Exception):
    """The operation is outside the approved scope."""
