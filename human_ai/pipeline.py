"""
human_ai/pipeline.py — STT → LLM → TTS pipeline orchestration.

Wires the three components together, manages the audio I/O tasks,
and runs the main turn loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ai_io import Flush, find_device, mic_capture, audio_player, _DISCARD
from config_loader import ProfileConfig
from human_ai.llm import LLM, TextChunk, ToolCallPending, TurnComplete, TurnError
from human_ai.stt import STT, INTERRUPT
from human_ai.tts import TTS

if TYPE_CHECKING:
    from tools import Tools

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the STT → LLM → TTS pipeline for one session.

    Usage::

        pipeline = Pipeline(cfg, tools, session_logger)
        await pipeline.run(stop_event)
    """

    def __init__(
        self,
        cfg: ProfileConfig,
        tools: Tools,
        session_logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.tools = tools
        self.logger = session_logger

        base_url: str = cfg.raw.get("base_url", "http://localhost:1234/v1")
        self.llm = LLM(base_url=base_url, config=cfg, tools=tools)
        self.stt = STT(config=cfg)
        self.tts = TTS(config=cfg)

        self._input_device = find_device(cfg.pipewire_sink_input, kind="input")
        self._output_device = find_device(cfg.pipewire_sink_output, kind="output")

        self._mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)
        self._stt_result_queue: asyncio.Queue = asyncio.Queue()
        self._audio_out_queue: asyncio.Queue = asyncio.Queue()

        # Set while the LLM/TTS is generating so STT can emit INTERRUPT.
        self._llm_generating = asyncio.Event()
        # The currently running LLM+TTS task, so it can be cancelled on interrupt.
        self._active_turn_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the pipeline until *stop_event* is set."""
        await self.llm.setup()  # also calls tools.setup()

        buffer_fill_bytes = int(
            self.cfg.receive_sample_rate * 2 * self.cfg.audio_buffer_seconds
        )

        mic_task = asyncio.create_task(
            mic_capture(
                self._mic_queue,
                self._input_device,
                self.cfg.send_sample_rate,
                self.cfg.chunk_size,
                stop_event,
            ),
            name="mic_capture",
        )
        player_task = asyncio.create_task(
            audio_player(
                self._audio_out_queue,
                self._output_device,
                self.cfg.receive_sample_rate,
                buffer_fill_bytes,
                self.cfg.buffer_clear_timeout_seconds,
            ),
            name="audio_player",
        )
        stt_task = asyncio.create_task(
            self.stt.run(self._mic_queue, self._stt_result_queue, self._llm_generating),
            name="stt",
        )

        self.logger.info("Session started. Listening...")
        silence_ping_interval: float = float(self.cfg.raw.get("silence_ping_interval", 5.0))
        last_activity = asyncio.get_event_loop().time()

        try:
            while not stop_event.is_set():
                try:
                    result = await asyncio.wait_for(
                        self._stt_result_queue.get(), timeout=0.2
                    )
                except asyncio.TimeoutError:
                    now = asyncio.get_event_loop().time()
                    if (
                        not self._llm_generating.is_set()
                        and now - last_activity >= silence_ping_interval
                    ):
                        last_activity = now
                        self.logger.debug("Silence ping → LLM")
                        asyncio.create_task(self._silence_ping(), name="silence_ping")
                    continue

                last_activity = asyncio.get_event_loop().time()

                if result is INTERRUPT:
                    self.logger.info("STT interrupt: stopping LLM and discarding audio buffer.")
                    self.llm.interrupt()
                    if self._active_turn_task and not self._active_turn_task.done():
                        self._active_turn_task.cancel()
                        try:
                            await self._active_turn_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    # Remove any dangling user message the LLM already added to history.
                    self.llm.rollback_user_turn()
                    # Drain the audio queue, then signal the player to abort.
                    while not self._audio_out_queue.empty():
                        try:
                            self._audio_out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self._audio_out_queue.put_nowait(_DISCARD)
                    continue

                if not isinstance(result, str) or not result.strip():
                    continue

                self.logger.info("STT → LLM: %r", result)
                self._active_turn_task = asyncio.create_task(
                    self._llm_tts_turn(result), name="llm_tts_turn"
                )
        finally:
            mic_task.cancel()
            stt_task.cancel()
            player_task.cancel()
            await self.llm.shutdown()  # also calls tools.teardown()
            await asyncio.gather(mic_task, stt_task, player_task, return_exceptions=True)

    # ------------------------------------------------------------------
    # Turn helpers
    # ------------------------------------------------------------------

    async def _llm_tts_turn(self, text: str, interrupt_first: bool = True) -> None:
        """
        Run one LLM turn with sequential TTS.

        The LLM streams into a work_queue.  Each TextChunk is synthesized and
        played to completion before the next item is processed.  ToolCallPending
        items are executed and their futures resolved so the LLM can continue.
        Everything is cancelled cleanly on asyncio.CancelledError (interrupt).
        """
        self._llm_generating.set()
        work_queue: asyncio.Queue = asyncio.Queue()

        producer = asyncio.create_task(
            # stream_work schedules as a sub-task internally; we await the
            # wrapper coroutine so cancellation propagates correctly.
            self._run_producer(text, work_queue, interrupt_first),
            name="llm_producer",
        )
        try:
            await self._process_work_queue(work_queue, producer)
        except asyncio.CancelledError:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            # Stop any audio currently playing.
            while not self._audio_out_queue.empty():
                try:
                    self._audio_out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._audio_out_queue.put_nowait(_DISCARD)
            raise
        finally:
            self._llm_generating.clear()

    async def _run_producer(
        self,
        text: str,
        work_queue: asyncio.Queue,
        interrupt_first: bool,
    ) -> None:
        """Thin wrapper: runs stream_work as a sub-task and cancels it on cancel."""
        task = self.llm.stream_work(text, work_queue, interrupt_first=interrupt_first)
        try:
            await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _process_work_queue(
        self,
        work_queue: asyncio.Queue,
        producer: asyncio.Task,
    ) -> None:
        """Consume work items sequentially until TurnComplete or error."""
        while True:
            item = await work_queue.get()

            if isinstance(item, TextChunk):
                if item.text.strip():
                    await self._speak_and_wait(item.text)

            elif isinstance(item, ToolCallPending):
                self.logger.info("Tool call: %s(%s)", item.fn_name, item.args)
                try:
                    if self.llm.tools is None:
                        raise RuntimeError("No tools registered")
                    result = await self.llm.tools.handle_call(item.fn_name, item.args)
                except Exception as exc:
                    result = f"Error: {exc}"
                    self.logger.error("Tool call '%s' failed: %s", item.fn_name, exc)
                item.result_future.set_result(result)

            elif isinstance(item, TurnComplete):
                self.llm.commit_response()
                break

            elif isinstance(item, TurnError):
                self.logger.error("LLM turn error: %s", item.exc)
                break

    async def _speak_and_wait(self, text: str) -> None:
        """Synthesize *text*, enqueue PCM, and block until the audio_player is done."""
        pcm_chunks = await self.tts.synthesize_one(text)
        for chunk in pcm_chunks:
            self._audio_out_queue.put_nowait(chunk)
        ack = asyncio.Event()
        self._audio_out_queue.put_nowait(Flush(ack=ack))
        await ack.wait()


    async def _llm_tts_turn_system(self, system_nudge: str) -> None:
        """Inject a system nudge without touching history."""
        self._active_turn_task = asyncio.current_task()
        self._llm_generating.set()
        try:
            token_queue: asyncio.Queue = asyncio.Queue()
            self.llm.generate_system(system_nudge, token_queue)
            await self.tts.run(token_queue, self._audio_out_queue)
            self.llm.commit_response()
        finally:
            self._llm_generating.clear()

    async def _silence_ping(self) -> None:
        """Fire a silent '...' turn if nothing else is generating."""
        self._active_turn_task = asyncio.current_task()
        if self._llm_generating.is_set():
            return
        await self._llm_tts_turn("...", interrupt_first=False)
