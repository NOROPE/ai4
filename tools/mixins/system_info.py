"""
tools/mixins/system_info.py — Example mixin: basic system information tools.

Demonstrates how to write a ToolMixin that Gemini can call.  Each
``@tool_function``-decorated async method becomes a callable tool
with an auto-generated FunctionDeclaration.
"""

from __future__ import annotations

import datetime
import os
import platform

from tools.base_mixin import ToolMixin, tool_function


class SystemInfoMixin(ToolMixin):
    """Provides simple system-information tools for Gemini."""

    MIXIN_NAME = "system_info"

    @tool_function(
        description="Get the current date and time in ISO-8601 format.",
    )
    async def get_current_time(self) -> str:
        return datetime.datetime.now().isoformat()

    @tool_function(
        description="Get basic information about the host system (OS, architecture, hostname).",
    )
    async def get_system_info(self) -> str:
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }
        return "\n".join(f"{k}: {v}" for k, v in info.items())

    @tool_function(
        description="Read the value of an environment variable.",
        parameter_descriptions={"name": "Name of the environment variable to read."},
    )
    async def get_env_var(self, name: str) -> str:
        value = os.environ.get(name)
        if value is None:
            return f"Environment variable '{name}' is not set."
        return value

    @tool_function(
        description="Evaluate a simple arithmetic expression and return the result.",
        parameter_descriptions={"expression": "A Python arithmetic expression (e.g. '2 + 3 * 4')."},
    )
    async def calculate(self, expression: str) -> str:
        # Restricted eval — only allow arithmetic on numbers
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: expression contains disallowed characters. Only digits and +-*/.() are permitted."
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"Error evaluating expression: {exc}"
