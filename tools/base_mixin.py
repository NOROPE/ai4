"""
tools/base_mixin.py — Base class and decorator for tool mixins.

Every mixin subclasses `ToolMixin` and decorates callable methods with
`@tool_function(...)`.  The `Tools` class introspects these decorators at
runtime to auto-generate OpenAI-compatible tool dicts so the LLM can call
them without any manual schema wiring.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine, get_type_hints

if TYPE_CHECKING:
    from config_loader import ProfileConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# @tool_function decorator — attaches schema metadata to a method
# ---------------------------------------------------------------------------

# Maps Python type annotations → JSON Schema type strings
_PY_TO_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class _ToolMeta:
    """Metadata stashed on a decorated method by @tool_function."""
    description: str
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    behavior: str | None = None


def tool_function(
    description: str,
    parameter_descriptions: dict[str, str] | None = None,
    behavior: str | None = None,
) -> Callable:
    """
    Mark a mixin method as a callable tool for the LLM.

    Usage::

        @tool_function(
            description="Add two numbers together",
            parameter_descriptions={"a": "First number", "b": "Second number"},
        )
        async def add(self, a: int, b: int) -> str:
            return str(a + b)

    The method **must** be ``async`` and return a ``str`` (the textual
    response sent back to the LLM).
    """
    def decorator(fn: Callable) -> Callable:
        fn._tool_meta = _ToolMeta(  # type: ignore[attr-defined]
            description=description,
            parameter_descriptions=parameter_descriptions or {},
            behavior=behavior,
        )
        return fn
    return decorator


def fire_and_forget(fn: Callable) -> Callable:
    """
    Make a tool method non-blocking: schedules the coroutine as a background
    ``asyncio.Task`` and returns an acknowledgement string immediately.

    Stack **inside** ``@tool_function``::

        @tool_function(description="Move the model")
        @fire_and_forget
        async def set_model_position(self, x: float, y: float) -> str:
            ...

    ``functools.wraps`` is used so that ``inspect.signature`` and
    ``get_type_hints`` still see the original parameter list.
    """
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        asyncio.create_task(fn(*args, **kwargs))
        return "OK"
    wrapper._is_fire_and_forget = True  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# ToolMixin base class
# ---------------------------------------------------------------------------


class ToolMixin:
    """
    Base class for all tool mixins.

    Subclasses should:
    1. Override `MIXIN_NAME` with a short identifier (used in config).
    2. Decorate async methods with `@tool_function(...)`.
    3. Optionally override `setup()` / `teardown()` for lifecycle hooks.
    """

    MIXIN_NAME: str = ""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def setup(self) -> None:
        """Called once after the mixin is instantiated. Override for init work."""

    async def teardown(self) -> None:
        """Called on shutdown. Override for cleanup."""

    # -- introspection helpers (used by Tools class) -------------------------

    def _get_tool_methods(self) -> list[tuple[str, Callable, _ToolMeta]]:
        """Return ``(name, bound_method, meta)`` for every @tool_function method."""
        results: list[tuple[str, Callable, _ToolMeta]] = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name, None)
            if callable(method) and hasattr(method, "_tool_meta"):
                results.append((attr_name, method, method._tool_meta))
        return results

    def _build_declarations(self) -> list[dict]:
        """Auto-generate OpenAI-compatible tool dicts from decorated methods."""
        declarations: list[dict] = []
        for name, method, meta in self._get_tool_methods():
            hints = get_type_hints(method)
            properties: dict[str, dict] = {}
            required: list[str] = []

            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                py_type = hints.get(param_name, str)
                schema_type = _PY_TO_SCHEMA.get(py_type, "string")
                prop: dict[str, Any] = {"type": schema_type}
                if param_name in meta.parameter_descriptions:
                    prop["description"] = meta.parameter_descriptions[param_name]
                properties[param_name] = prop
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            parameters: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            }
            if required:
                parameters["required"] = required

            declarations.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.description,
                    "parameters": parameters,
                },
            })
        return declarations
