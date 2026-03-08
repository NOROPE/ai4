from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import APIError, AsyncOpenAI
from openai._types import NOT_GIVEN

from config_loader import ProfileConfig

if TYPE_CHECKING:
    from tools import Tools

logger = logging.getLogger(__name__)

# Sentinel placed on the token queue to signal end-of-response.
_END = object()

# The AI can output this exact string (alone) to stay silent for a turn.
# It will be stripped from TTS and removed from conversation history.
SILENT_MARKER = "[STAY]"

# Characters that end a speakable sentence chunk.
_SENTENCE_BOUNDARIES = frozenset(".!?\n")

# Regex to strip complete <think>…</think> blocks (for post-hoc cleanup in history).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Work-queue item types  (used by stream_work / pipeline work processor)
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """A complete sentence ready to synthesize and speak."""
    text: str

@dataclass
class ToolCallPending:
    """A tool call that must be executed before LLM can continue."""
    call_id: str
    fn_name: str
    args: dict
    result_future: asyncio.Future = field(repr=False)

class TurnComplete:
    """Signals that the LLM turn finished normally."""

class TurnError:
    """Signals that the LLM turn encountered an unrecoverable error."""
    def __init__(self, exc: Exception) -> None:
        self.exc = exc


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
        # Holds the last assistant response until commit_response() is called
        # (after TTS finishes) so history only reflects what was actually spoken.
        self._pending_content: str | None = None

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

    def _build_system_content(self) -> str:
        """Return the system instruction, prepending any important memories."""
        base = self.config.system_instruction
        if self.tools:
            mem_mixin = self.tools.get_mixin("memories")
            if mem_mixin is not None:
                getter = getattr(mem_mixin, "get_important_memories_text", None)
                if callable(getter):
                    important_text = str(getter())
                if important_text:
                    return important_text + "\n\n" + base
        return base

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
        self._pending_content = None

    def rollback_user_turn(self) -> None:
        """Remove the last user message from history if no assistant reply was committed.

        Call this after cancelling a turn so the next user message doesn't produce
        two consecutive user messages, which breaks model chat templates.
        """
        if self._history and self._history[-1].get("role") == "user":
            self._history.pop()
            logger.debug("Rolled back dangling user message from history.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str,
        token_queue: asyncio.Queue,
        interrupt_first: bool = True,
    ) -> asyncio.Task:
        """
        Schedule streaming generation and return the Task.

        Tokens are put on *token_queue* as ``str`` values.
        ``_END`` is put when the turn completes; ``None`` on error/cancel.
        Set *interrupt_first=False* to skip cancelling any in-flight request
        (e.g. for low-priority background pings that should not clobber
        an ongoing response).
        """
        if interrupt_first:
            self.interrupt()  # cancel any prior in-flight request
        elif self._task and not self._task.done():
            # Another turn is already running — skip this one entirely.
            logger.debug("generate(interrupt_first=False): skipping, task already in flight.")
            asyncio.get_event_loop().call_soon(lambda: token_queue.put_nowait(_END))
            return self._task
        self._task = asyncio.create_task(
            self._generate(user_message, token_queue),
            name="llm_generate",
        )
        return self._task

    def generate_system(
        self,
        system_nudge: str,
        token_queue: asyncio.Queue,
    ) -> asyncio.Task:
        """
        Like generate(), but injects *system_nudge* as an extra system message
        rather than a user turn.  Nothing is added to persistent history.
        """
        self.interrupt()
        self._task = asyncio.create_task(
            self._generate_system(system_nudge, token_queue),
            name="llm_generate_system",
        )
        return self._task

    # ------------------------------------------------------------------
    # Work-queue generation  (new sequential pipeline)
    # ------------------------------------------------------------------

    def stream_work(
        self,
        user_message: str,
        work_queue: asyncio.Queue,
        interrupt_first: bool = True,
    ) -> asyncio.Task:
        """
        Schedule streaming generation into *work_queue* and return the Task.

        Items put on *work_queue*:
          TextChunk        — a complete sentence to speak
          ToolCallPending  — a tool call; set its result_future to unblock LLM
          TurnComplete     — normal end of turn
          TurnError        — unrecoverable error
        """
        if interrupt_first:
            self.interrupt()
        elif self._task and not self._task.done():
            logger.debug("stream_work(interrupt_first=False): skipping, task in flight.")
            asyncio.get_event_loop().call_soon(lambda: work_queue.put_nowait(TurnComplete()))
            return self._task
        self._task = asyncio.create_task(
            self._do_stream_work(user_message, work_queue),
            name="llm_stream_work",
        )
        return self._task

    async def _do_stream_work(
        self,
        user_message: str,
        work_queue: asyncio.Queue,
    ) -> None:
        """Add user message, build context, run _stream_work_loop."""
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        system_msg: dict[str, Any] = {
            "role": "system",
            "name": self.config.name,
            "content": self._build_system_content(),
        }
        messages: list[dict[str, Any]] = [system_msg] + list(self._history)

        try:
            await self._stream_work_loop(messages, work_queue)
        except asyncio.CancelledError:
            logger.info("LLM stream_work cancelled.")
            if self._history and self._history[-1].get("role") == "user":
                self._history.pop()
            raise
        except APIError as exc:
            if "context" in str(exc).lower():
                # Drop the oldest half of history and retry once.
                self._history = self._history[len(self._history) // 2:]
                while self._history and self._history[0].get("role") != "user":
                    self._history.pop(0)
                logger.warning(
                    "Context size exceeded — trimmed history to %d messages, retrying.",
                    len(self._history),
                )
                messages = [system_msg] + list(self._history)
                try:
                    await self._stream_work_loop(messages, work_queue)
                except Exception as retry_exc:
                    logger.error("LLM stream_work retry error: %s", retry_exc, exc_info=True)
                    work_queue.put_nowait(TurnError(retry_exc))
            else:
                logger.error("LLM stream_work API error: %s", exc, exc_info=True)
                work_queue.put_nowait(TurnError(exc))
        except Exception as exc:
            logger.error("LLM stream_work error: %s", exc, exc_info=True)
            work_queue.put_nowait(TurnError(exc))

    async def _stream_work_loop(
        self,
        messages: list[dict[str, Any]],
        work_queue: asyncio.Queue,
    ) -> None:
        """
        Streaming loop that emits structured work items instead of raw tokens.

        Text is split into sentences at .!?\\n boundaries.
        Tool calls are emitted as ToolCallPending; the loop awaits their futures
        before making the follow-up API call.
        """
        while True:
            tool_config = (
                self.tools.get_tool_config()
                if self.tools and self.tools.has_tools
                else NOT_GIVEN
            )

            extra_kwargs: dict[str, Any] = {}
            if tool_config is not NOT_GIVEN:
                extra_kwargs["tool_choice"] = "auto"
                extra_kwargs["parallel_tool_calls"] = False

            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tool_config,  # type: ignore[arg-type]
                stream=True,
                **extra_kwargs,
            )

            accumulated_content = ""
            text_buf = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            _in_think = False  # True while inside a <think>…</think> block

            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                # --- text tokens: split into sentences on the fly ---
                if delta.content:
                    accumulated_content += delta.content

                    # Filter out <think>…</think> reasoning blocks so they are
                    # never spoken aloud.  Tags may span multiple chunks.
                    chunk_text = delta.content
                    filtered_text = ""
                    while chunk_text:
                        if not _in_think:
                            think_open = chunk_text.find("<think>")
                            think_close = chunk_text.find("</think>")
                            # Stray </think> before any <think>: skip it.
                            if think_close != -1 and (think_open == -1 or think_close < think_open):
                                filtered_text += chunk_text[:think_close]
                                chunk_text = chunk_text[think_close + 8:]  # skip "</think>"
                                continue
                            if think_open == -1:
                                filtered_text += chunk_text
                                break
                            filtered_text += chunk_text[:think_open]
                            chunk_text = chunk_text[think_open + 7:]  # skip "<think>"
                            _in_think = True
                        else:
                            idx = chunk_text.find("</think>")
                            if idx == -1:
                                break  # rest of chunk is think content — discard
                            chunk_text = chunk_text[idx + 8:]  # skip "</think>"
                            _in_think = False

                    for ch in filtered_text:
                        text_buf += ch
                        if ch in _SENTENCE_BOUNDARIES and text_buf.strip():
                            sentence = text_buf.strip().replace(SILENT_MARKER, "").strip()
                            text_buf = ""
                            if sentence:
                                await work_queue.put(TextChunk(sentence))

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

            # Flush any leftover text that didn't end with a boundary.
            remaining = text_buf.strip().replace(SILENT_MARKER, "").strip()
            if remaining:
                await work_queue.put(TextChunk(remaining))

            logger.debug(
                "LLM stream done — finish_reason=%r, tool_calls=%d, content_len=%d",
                finish_reason, len(tool_calls_acc), len(accumulated_content),
            )

            # Strip any <think>…</think> blocks from accumulated content before
            # saving to history or checking for SILENT_MARKER.
            accumulated_content = _THINK_RE.sub("", accumulated_content).strip()

            # Build the assistant message for history.
            ai_msg: dict[str, Any] = {
                "role": "assistant",
                "name": self.config.name,
                "content": accumulated_content or None,
            }

            # --- handle tool calls ---
            if tool_calls_acc and self.tools:
                ai_msg["tool_calls"] = list(tool_calls_acc.values())
                messages.append(ai_msg)

                # Emit one ToolCallPending per call; await all futures in parallel.
                loop = asyncio.get_running_loop()
                pending: list[tuple[str, asyncio.Future]] = []
                for tc in tool_calls_acc.values():
                    fn_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                        logger.warning("Could not parse tool args for %s", fn_name)
                    fut: asyncio.Future = loop.create_future()
                    await work_queue.put(
                        ToolCallPending(
                            call_id=tc["id"],
                            fn_name=fn_name,
                            args=args,
                            result_future=fut,
                        )
                    )
                    pending.append((tc["id"], fut))

                # Block until the work processor resolves every future.
                results = await asyncio.gather(*[f for _, f in pending])

                for (call_id, _), result in zip(pending, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    })
                continue  # next LLM call with tool results appended

            # --- turn complete (no tool calls) ---
            messages.append(ai_msg)

            if accumulated_content.strip() == SILENT_MARKER:
                if self._history and self._history[-1].get("role") == "user":
                    self._history.pop()
                logger.debug("LLM: silent turn (no response).")
                self._pending_content = None
            elif accumulated_content:
                self._pending_content = accumulated_content

            work_queue.put_nowait(TurnComplete())
            break

    # ------------------------------------------------------------------
    # Legacy token-queue generation  (kept for silence pings / system nudges)
    # ------------------------------------------------------------------

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
            "content": self._build_system_content(),
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

    async def _generate_system(
        self,
        system_nudge: str,
        token_queue: asyncio.Queue,
    ) -> None:
        """Internal coroutine: inject a system nudge as a user turn.

        The nudge and the AI's reply are saved to history as a proper
        user/assistant pair so subsequent calls always see valid alternating
        roles.  On cancel or error the history is restored to its prior state.
        """
        system_msg: dict[str, Any] = {
            "role": "system",
            "name": self.config.name,
            "content": self._build_system_content(),
        }

        # Add the nudge to history first (mirrors what _generate does for user
        # messages) so _stream_loop's assistant-save produces a proper pair.
        _history_snapshot = list(self._history)
        self._history.append({"role": "user", "content": system_nudge})
        self._trim_history()

        messages: list[dict[str, Any]] = [system_msg] + list(self._history)

        try:
            await self._stream_loop(messages, token_queue)
        except asyncio.CancelledError:
            logger.info("LLM system generate cancelled.")
            self._history = _history_snapshot
            await token_queue.put(None)
            raise
        except Exception as exc:
            logger.error("LLM system generate error: %s", exc, exc_info=True)
            self._history = _history_snapshot
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

            extra_kwargs: dict[str, Any] = {}
            if tool_config is not NOT_GIVEN:
                extra_kwargs["tool_choice"] = "auto"
                extra_kwargs["parallel_tool_calls"] = False

            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tool_config,  # type: ignore[arg-type]
                stream=True,
                **extra_kwargs,
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
            # Some OpenAI-compatible servers (e.g. LM Studio) return finish_reason
            # "stop" even when tool calls are present, so check tool_calls_acc alone.
            if tool_calls_acc and self.tools:
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
            if accumulated_content.strip() == SILENT_MARKER:
                # AI chose not to respond — remove the unanswered user message
                # from history so the exchange is invisible to future context.
                if self._history and self._history[-1].get("role") == "user":
                    self._history.pop()
                logger.debug("LLM: silent turn (no response).")
                self._pending_content = None
            elif accumulated_content:
                # Stage the response — commit_response() saves it after TTS speaks.
                self._pending_content = accumulated_content
            await token_queue.put(_END)
            break

    def commit_response(self) -> None:
        """Save the last assistant response to history.

        Call this after TTS finishes speaking so history only reflects
        what the user has actually heard.
        """
        if self._pending_content:
            self._history.append({"role": "assistant", "content": self._pending_content})
            self._trim_history()
            self._save_history()
            self._pending_content = None

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Keep at most ``prevmsg_count`` messages in memory.

        Always trims to a user-first boundary so we never send
        [system, assistant, ...] which confuses model prompt templates.
        """
        limit = self.config.prevmsg_count
        if len(self._history) > limit:
            self._history = self._history[-limit:]
        # Drop leading assistant/tool messages so history always starts with user.
        while self._history and self._history[0].get("role") != "user":
            self._history.pop(0)

    def clear_history(self) -> None:
        """Wipe in-memory conversation history."""
        self._history.clear()

    @property
    def end_token(self) -> object:
        """The sentinel value placed on the token queue at end-of-turn."""
        return _END

