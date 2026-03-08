"""
human_ai/tts.py — Text-to-speech component.

Reads text tokens from an asyncio.Queue produced by the LLM, accumulates
them into sentence-length chunks, synthesizes audio, and forwards PCM bytes
to the audio_player queue.  Sends ai_io._FLUSH when the token stream ends.

TODO: plug in a real TTS backend (Kokoro, Azure TTS, Piper, …).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from kokoro import KPipeline
from ai_io import _FLUSH

logger = logging.getLogger(__name__)

# Characters that mark a natural sentence boundary for chunking.
_BOUNDARIES = frozenset(".!?\n")


class TTS:
    """
    Text-to-speech component.

    Call ``await tts.run(token_queue, audio_queue)`` once per LLM turn.
    It drains *token_queue* until a non-``str`` sentinel appears (either
    ``LLM._END`` or ``None`` on error), synthesizes each sentence chunk,
    and puts raw PCM bytes on *audio_queue*.  When the turn is done it puts
    ``_FLUSH`` so the audio_player plays any remaining buffered audio.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.TTS")
        self.pipeline = KPipeline(lang_code='a')
        

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        token_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
    ) -> None:
        """
        Stream tokens from *token_queue* → synthesize → *audio_queue*.
        Non-str token signals end-of-turn; sends _FLUSH to audio_queue.
        """
        pending = ""
        try:
            while True:
                token = await token_queue.get()
                if not isinstance(token, str):
                    # End-of-turn sentinel (LLM._END or None on error)
                    if pending.strip():
                        await self._synthesize(pending.strip(), audio_queue)
                    audio_queue.put_nowait(_FLUSH)
                    return
                pending += token
                if pending[-1] in _BOUNDARIES and pending.strip():
                    await self._synthesize(pending.strip(), audio_queue)
                    pending = ""
        except asyncio.CancelledError:
            audio_queue.put_nowait(_FLUSH)
            raise

    # ------------------------------------------------------------------
    # Synthesis backend (stub — replace with real TTS call)
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str, audio_queue: asyncio.Queue) -> None:
        """
        Convert *text* to speech and put PCM byte chunks on *audio_queue*.

        TODO: call a real TTS API here and put the returned PCM bytes on
        audio_queue.  Example sketch::

            pcm_bytes = await my_tts_client.synthesize(text, sample_rate=24000)
            audio_queue.put_nowait(pcm_bytes)
        """

        generator = self.pipeline(
            text, voice=self.config.voice,
            speed=1, split_pattern=r'\n+'
        )
        self._logger.info("TTS (stub): %s", text)
