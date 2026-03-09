


from tools.base_mixin import ToolMixin, tool_function
from dataclasses import dataclass, field
from typing import Callable, get_type_hints
import inspect
import openai


# agent tool function decorator
@dataclass
class _ToolMeta:
    """Metadata stashed on a decorated method by @tool_function."""
    description: str
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    behavior: str | None = None


def agent_tool_function(
    description: str,
    parameter_descriptions: dict[str, str] | None = None,
    behavior: str | None = None,
) -> Callable:
    """
    Mark a mixin method as a callable tool for the LLM's subagent.
    Usage::

        @agent_tool_function(
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

class AgentModeMixin(ToolMixin):
    """
    A subclass of ToolMixin that adds agent mode functionality
    Override class attributes AGENT_DESC and PARAM_DESC to set custom descriptions for the agent. (by setting the class attributes)
    call setup_agent() in setup() to initialize the agent.
    call teardown_agent() in teardown() to clean up the agent.
    """
    
    AGENT_DESC = "Base description"
    PARAM_DESC = "Base parameter info"
    
    def __init__(self, config):
        super().__init__(config)
        self.agent = None # this is bad practice
        self.agent_config = config.raw.get("subagent_config", {})
        self.agent_name = self.agent_config.get("subagent_name", "")
        self.agent_url = self.agent_config.get("subagent_url", "")
    async def setup_agent(self) -> None:
        
        
    async def teardown_agent(self) -> None:
        print("teardown") 
        
    @tool_function(
            description="Placeholder",  # This will be overwritten in __init__
            parameter_descriptions={"what_to_do": "Placeholder"}
        )
    async def do(self, what_to_do: str) -> str:
            # Your implementation here
            return f"Executed: {what_to_do}"
            
    # -- introspection helpers (used by agent) -------------------------

    def _get_agent_tool_methods(self) -> list[tuple[str, Callable, _ToolMeta]]:
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

            