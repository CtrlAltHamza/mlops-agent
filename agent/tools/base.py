"""Tool execution wrapper: timeouts, structured results, and logs."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from agent.logging_utils import log_tool_event
from agent.schemas import (
    ConfirmationRequired,
    InvalidRequest,
    NotAuthorized,
    ToolResult,
    ToolUnavailable,
)

ToolFn = Callable[..., dict[str, Any]]


class Tool:
    def __init__(self, name: str, fn: ToolFn, timeout_s: float = 15.0) -> None:
        self.name = name
        self.fn = fn
        self.timeout_s = timeout_s


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        request_id: str,
        timeout_s: float | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        arguments = arguments or {}
        if name not in self._tools:
            result = ToolResult(
                ok=False,
                tool=name,
                error="unknown_tool",
                status="rejected",
                elapsed_ms=_elapsed(started),
            )
            log_tool_event(
                request_id=request_id,
                tool=name,
                status=result.status,
                duration_ms=result.elapsed_ms,
                error=result.error,
            )
            return result

        tool = self._tools[name]
        limit = timeout_s if timeout_s is not None else tool.timeout_s
        status = "ok"
        error: str | None = None
        data: dict[str, Any] | None = None
        try:
            data = _call(tool.fn, arguments, limit)
        except FuturesTimeout:
            status = "timeout"
            error = "timeout"
        except InvalidRequest as exc:
            status = "invalid_request"
            error = str(exc)
        except ConfirmationRequired as exc:
            status = "confirmation_required"
            error = str(exc)
        except NotAuthorized as exc:
            status = "rejected"
            error = str(exc)
        except (ToolUnavailable, FileNotFoundError, ConnectionError, TimeoutError) as exc:
            status = "unavailable"
            error = str(exc) or "unavailable"
        except PermissionError as exc:
            message = str(exc)
            if message.startswith("confirmation_required"):
                status = "confirmation_required"
            else:
                status = "rejected"
            error = message
        except Exception as exc:  # noqa: BLE001 - external systems fail in many ways
            status = "unavailable"
            error = f"{type(exc).__name__}: {exc}"
        result = ToolResult(
            ok=status == "ok",
            tool=name,
            data=data if status == "ok" else None,
            error=error,
            status=status,
            elapsed_ms=_elapsed(started),
        )
        log_tool_event(
            request_id=request_id,
            tool=name,
            status=result.status,
            duration_ms=result.elapsed_ms,
            error=result.error,
        )
        return result


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _call(fn: ToolFn, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, **arguments)
        return future.result(timeout=timeout_s)
