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
import scipy.signal as sps
import numpy as np
import torch
from kokoro import KPipeline
from ai_io import _FLUSH
from config_loader import ProfileConfig
from human_ai.llm import SILENT_MARKER

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

    def __init__(self, config: ProfileConfig) -> None:
        self.config = config
        self.voice = config.voice
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
                    speakable = pending.strip().replace(SILENT_MARKER, "").strip()
                    if speakable:
                        await self._synthesize(speakable, audio_queue)
                    audio_queue.put_nowait(_FLUSH)
                    return
                pending += token
                if pending[-1] in _BOUNDARIES and pending.strip():
                    speakable = pending.strip().replace(SILENT_MARKER, "").strip()
                    if speakable:
                        await self._synthesize(speakable, audio_queue)
                    pending = ""
        except asyncio.CancelledError:
            audio_queue.put_nowait(_FLUSH)
            raise

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str, audio_queue: asyncio.Queue) -> None:
        """Convert *text* to speech using Kokoro and put PCM chunks on *audio_queue*."""
        chunks = await self.synthesize_one(text)
        for chunk in chunks:
            audio_queue.put_nowait(chunk)

    async def synthesize_one(self, text: str) -> list[bytes]:
        """Synthesize *text* and return raw int16 PCM chunks (no queue side-effects)."""
        self._logger.info("TTS: %s", text)
        target_sr: int = getattr(self.config, "receive_sample_rate", 24000)
        voice = self.voice

        def _run_kokoro() -> list[bytes]:
            chunks: list[bytes] = []
            for _, _, audio in self.pipeline(
                text,
                voice=voice,
                speed=1.0,
                split_pattern=r'\n+',
            ):
                if audio is None or len(audio) == 0:
                    continue
                samples: np.ndarray = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
                if target_sr != 24000:
                    num_samples = int(round(len(samples) * target_sr / 24000))
                    samples = np.array(sps.resample(samples, num_samples), dtype=np.float32)
                pcm: bytes = np.array(np.clip(samples, -1.0, 1.0) * 32767, dtype=np.int16).tobytes()
                chunks.append(pcm)
            return chunks

        return await asyncio.to_thread(_run_kokoro)


        