from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

from config_loader import ProfileConfig

if TYPE_CHECKING:
    from tools import Tools

logger = logging.getLogger(__name__)

# Sentinel placed on the token queue to signal end-of-response.
_END = object()


class LLM:
    """
    Async OpenAI-compatible LLM wrapper.

    1. Maintains per-session conversation history, persisted to
       ``logs/<profile>/transcriptions.json`` (last ``prevmsg_count`` messages).
    2. Streams tokens into ``token_queue`` as they arrive; a tool call mid-
       stream is handled transparently and generation resumes automatically.
    3. ``token_queue`` receives ``str`` chunks then the ``_END`` sentinel when
       a full turn is done (or ``None`` on interrupt/error).
    4. Clean shutdown via ``interrupt()`` (cancels current stream) and
       ``shutdown()`` (tears down tools).
    """

    def __init__(
        self,
        base_url: str,
        config: ProfileConfig,
        tools: Tools | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=config.raw.get("api_key", "none"),
            **config.raw.get("client_args", {}),
        )
        self.config = config
        self.tools = tools

        # Persistent history file: logs/<profile>/transcriptions.json
        self._history_path = config.logs_dir / "transcriptions.json"

        # In-memory message history (excludes system message).
        self._history: list[dict[str, Any]] = []
        self._load_history()

        # Track the running generate task so it can be interrupted.
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        """Load the last ``prevmsg_count`` messages from the JSON history file."""
        if not self._history_path.exists():
            return
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._history = data[-self.config.prevmsg_count:]
                logger.debug("Loaded %d messages from %s", len(self._history), self._history_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load history from %s: %s", self._history_path, exc)

    def _save_history(self) -> None:
        """Persist the current in-memory history to the JSON history file."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save history to %s: %s", self._history_path, exc)

    async def setup(self) -> None:
        """Initialise tools (if any). Call before the first generate()."""
        if self.tools:
            await self.tools.setup()

    async def shutdown(self) -> None:
        """Cancel any active stream and tear down tools."""
        self.interrupt()
        if self.tools:
            await self.tools.teardown()

    # ------------------------------------------------------------------
    # Interrupt
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Cancel the currently running generate() task, if any."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("LLM stream interrupted.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str,
        token_queue: asyncio.Queue,
    ) -> asyncio.Task:
        """
        Schedule streaming generation and return the Task.

        Tokens are put on *token_queue* as ``str`` values.
        ``_END`` is put when the turn completes; ``None`` on error/cancel.
        """
        self.interrupt()  # cancel any prior in-flight request
        self._task = asyncio.create_task(
            self._generate(user_message, token_queue),
            name="llm_generate",
        )
        return self._task

    async def _generate(
        self,
        user_message: str,
        token_queue: asyncio.Queue,
    ) -> None:
        """Internal coroutine: stream, handle tool calls, recurse until done."""
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        system_msg: dict[str, Any] = {
            "role": "system",
            "name": self.config.name,
            "content": self.config.system_instruction,
        }
        messages: list[dict[str, Any]] = [system_msg] + list(self._history)

        try:
            await self._stream_loop(messages, token_queue)
        except asyncio.CancelledError:
            logger.info("LLM generate cancelled.")
            # Remove the unanswered user message so history stays consistent
            # this prevents interrrupted messagtes from continuing later
            if self._history and self._history[-1].get("role") == "user":
                self._history.pop()
            await token_queue.put(None)
            raise
        except Exception as exc:
            logger.error("LLM generate error: %s", exc, exc_info=True)
            await token_queue.put(None)

    async def _stream_loop(
        self,
        messages: list[dict[str, Any]],
        token_queue: asyncio.Queue,
    ) -> None:
        """
        Repeatedly call the model until no more tool calls remain.
        Tokens are forwarded to *token_queue* in real-time.
        """
        while True:
            tool_config = (
                self.tools.get_tool_config()
                if self.tools and self.tools.has_tools
                else NOT_GIVEN
            )

            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tool_config,  # type: ignore[arg-type]
                stream=True,
            )

            accumulated_content = ""
            # index → partial tool-call dict
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                # --- text tokens ---
                if delta.content:
                    accumulated_content += delta.content
                    await token_queue.put(delta.content)

                # --- tool-call fragments ---
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

            # Build the assistant turn for history / next request
            ai_msg: dict[str, Any] = {
                "role": "assistant",
                "name": self.config.name,
                "content": accumulated_content or None,
            }
            if tool_calls_acc:
                ai_msg["tool_calls"] = list(tool_calls_acc.values())
            messages.append(ai_msg)

            # --- handle tool calls ---
            if finish_reason == "tool_calls" and tool_calls_acc and self.tools:
                for tc in tool_calls_acc.values():
                    fn_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                        logger.warning("Could not parse tool args for %s", fn_name)

                    result = await self.tools.handle_call(fn_name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                # Loop back to let the model produce its final reply
                continue

            # --- turn complete ---
            if accumulated_content:
                # Keep only the assistant's final text in persistent history
                self._history.append({"role": "assistant", "content": accumulated_content})
                self._trim_history()
                self._save_history()
            await token_queue.put(_END)
            break

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Keep at most ``prevmsg_count`` messages in memory."""
        limit = self.config.prevmsg_count
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def clear_history(self) -> None:
        """Wipe in-memory conversation history."""
        self._history.clear()

    @property
    def end_token(self) -> object:
        """The sentinel value placed on the token queue at end-of-turn."""
        return _END

