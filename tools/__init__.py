"""
tools/__init__.py — Tools orchestrator.

The `Tools` class dynamically composes `ToolMixin` subclasses based on the
profile's ``tool_mixins`` list.  It:

* Discovers and instantiates the requested mixins.
* Collects their ``@tool_function``-decorated methods.
* Auto-generates ``google.genai.types.FunctionDeclaration`` objects for
  Gemini's ``LiveConnectConfig``.
* Dispatches incoming tool calls to the correct mixin method.
nd implement them in the `tools/mixins/` directory.
Make sure to add new mixins to the `MIXIN_REGISTRY` below, a
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

from google.genai import types

from tools.base_mixin import ToolMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mixin registry — maps MIXIN_NAME → module path
# ---------------------------------------------------------------------------
# Add new mixins here.  The value is the dotted import path of the module
# that contains the ToolMixin subclass.
MIXIN_REGISTRY: dict[str, str] = {
    "system_info": "tools.mixins.system_info",
    "live2d": "tools.mixins.live2d_mixin",
}


def _load_mixin_class(name: str) -> type[ToolMixin]:
    """Import the module for *name* and return the ToolMixin subclass inside it."""
    module_path = MIXIN_REGISTRY.get(name)
    if module_path is None:
        raise ValueError(
            f"Unknown tool mixin '{name}'. "
            f"Available: {sorted(MIXIN_REGISTRY.keys())}"
        )
    module = importlib.import_module(module_path)
    # Find the first ToolMixin subclass in the module
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ToolMixin)
            and obj is not ToolMixin
            and getattr(obj, "MIXIN_NAME", "") == name
        ):
            return obj
    raise ValueError(
        f"Module '{module_path}' has no ToolMixin subclass with MIXIN_NAME='{name}'."
    )


# ---------------------------------------------------------------------------
# Tools class
# ---------------------------------------------------------------------------


class Tools:
    """
    Async-capable tool manager composed from profile-selected mixins.

    Usage::

        tools = Tools(cfg.tool_mixins)
        await tools.setup()

        # Pass declarations into Gemini config
        live_config = types.LiveConnectConfig(
            tools=[tools.get_tool_config()],
            ...
        )

        # When Gemini returns a tool call:
        result = await tools.handle_call(function_name, args_dict)
    """

    def __init__(self, mixin_names: list[str], config: Any = None, **mixin_kwargs: Any) -> None:
        self._mixin_names = mixin_names
        self._config = config
        self._mixin_kwargs = mixin_kwargs  # e.g. volume_queue=..., tts_queue=...
        self._mixins: list[ToolMixin] = []
        # name → (bound_method, meta) — populated in setup()
        self._dispatch: dict[str, Any] = {}
        self._declarations: list[types.FunctionDeclaration] = []

    # -- lifecycle -----------------------------------------------------------

    async def setup(self) -> None:
        """Instantiate all requested mixins and build the dispatch table."""
        for name in self._mixin_names:
            cls = _load_mixin_class(name)
            instance = cls(config=self._config, **self._mixin_kwargs)
            await instance.setup()
            self._mixins.append(instance)
            logger.info("Loaded tool mixin: %s (%s)", name, cls.__name__)

            for fn_name, method, meta in instance._get_tool_methods():
                if fn_name in self._dispatch:
                    logger.warning(
                        "Duplicate tool function '%s' from mixin '%s' — overwriting.",
                        fn_name, name,
                    )
                self._dispatch[fn_name] = method

            self._declarations.extend(instance._build_declarations())

        logger.info(
            "Tools ready — %d mixin(s), %d function(s): %s",
            len(self._mixins),
            len(self._declarations),
            [d.name for d in self._declarations],
        )

    async def teardown(self) -> None:
        """Shut down all mixins (reverse order)."""
        for mixin in reversed(self._mixins):
            try:
                await mixin.teardown()
            except Exception as exc:
                logger.warning("Mixin teardown error (%s): %s", mixin.MIXIN_NAME, exc)

    # -- Gemini integration --------------------------------------------------

    def get_declarations(self) -> list[types.FunctionDeclaration]:
        """Return the list of FunctionDeclarations for Gemini config."""
        return list(self._declarations)

    def get_tool_config(self) -> types.Tool:
        """Return a ``types.Tool`` ready to pass into ``LiveConnectConfig.tools``."""
        return types.Tool(function_declarations=self._declarations)

    @property
    def has_tools(self) -> bool:
        """True if at least one tool function is registered."""
        return bool(self._declarations)

    # -- dispatch ------------------------------------------------------------

    async def handle_call(self, function_name: str, args: dict[str, Any]) -> str:
        """
        Dispatch a tool call from Gemini to the appropriate mixin method.

        Returns the string result to send back to Gemini.
        """
        method = self._dispatch.get(function_name)
        if method is None:
            msg = f"Unknown tool function: '{function_name}'"
            logger.error(msg)
            return msg
        try:
            logger.info("Tool call: %s(%s)", function_name, args)
            result = await method(**args)
            logger.info("Tool result: %s → %s", function_name, result[:200] if len(result) > 200 else result)
            return result
        except Exception as exc:
            logger.error("Tool call '%s' failed: %s", function_name, exc)
            return f"Error executing {function_name}: {exc}"
