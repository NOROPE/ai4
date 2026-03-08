"""
human_ai/stt.py — Speech-to-text component.

Interface
---------
  result_queue receives:
    str          — completed utterance text, ready to send to the LLM
    INTERRUPT    — user started speaking while the LLM/TTS is still active;
                   the main loop should interrupt the LLM and discard buffered
                   TTS audio before forwarding the next utterance

VAD: simple energy-threshold VAD.
  Speech starts when RMS > stt_volume_threshold.
  Speech ends after stt_silence_duration seconds below threshold.

ASR: faster-whisper.
  Model size controlled by config key ``stt_model`` (default "base").
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any

import numpy as np
from faster_whisper import WhisperModel  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Sentinel — put on result_queue when speech is detected mid-response.
INTERRUPT = object()

# VAD / ASR defaults (overridable via config.raw)
_DEFAULT_VOLUME_THRESHOLD = 500       # RMS of int16 samples (0–32768)
_DEFAULT_SILENCE_DURATION = 0.8       # seconds of quiet to end an utterance
_DEFAULT_MIN_SPEECH_DURATION = 0.3    # seconds; shorter clips are discarded
_DEFAULT_MODEL = "base"               # faster-whisper model name
_DEFAULT_DEVICE = "cpu"
_DEFAULT_COMPUTE_TYPE = "int8"


def _rms(pcm_bytes: bytes) -> float:
    """Return RMS amplitude of a raw int16 PCM chunk."""
    if not pcm_bytes:
        return 0.0
    count = len(pcm_bytes) // 2
    samples = struct.unpack_from(f"{count}h", pcm_bytes)
    return float(np.sqrt(np.mean(np.square(np.array(samples, dtype=np.float32)))))


class STT:
    """
    Speech-to-text component.

    Reads raw PCM audio (int16, mono) from *mic_queue* and emits results on
    *result_queue*:
      - ``str``       — a completed utterance to send to the LLM
      - ``INTERRUPT`` — new speech detected while the LLM is still generating;
                        the caller should interrupt the LLM and discard
                        buffered audio before the next utterance arrives
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.STT")

        raw = getattr(config, "raw", {}) or {}
        self._volume_threshold: float = float(raw.get("stt_volume_threshold", _DEFAULT_VOLUME_THRESHOLD))
        self._silence_duration: float = float(raw.get("stt_silence_duration", _DEFAULT_SILENCE_DURATION))
        self._min_speech_duration: float = float(raw.get("stt_min_speech_duration", _DEFAULT_MIN_SPEECH_DURATION))
        self._model_name: str = str(raw.get("stt_model", _DEFAULT_MODEL))
        self._device: str = str(raw.get("stt_device", _DEFAULT_DEVICE))
        self._compute_type: str = str(raw.get("stt_compute_type", _DEFAULT_COMPUTE_TYPE))
        self._sample_rate: int = getattr(config, "send_sample_rate", 16000)

        self._model: WhisperModel | None = None  # lazy-loaded on first transcription

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        mic_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
        llm_generating: asyncio.Event | None = None,
    ) -> None:
        """
        Main loop:
        1. Read PCM chunks from *mic_queue*.
        2. Run energy-threshold VAD: accumulate frames while RMS > threshold,
           end segment after *silence_duration* seconds of quiet.
        3. Transcribe the segment with faster-whisper.
        4. Emit INTERRUPT on *result_queue* if *llm_generating* is set and
           speech was detected, then emit the transcribed string.
        """
        speech_buffer: list[bytes] = []
        in_speech = False
        last_speech_time: float = 0.0
        interrupt_sent = False

        self._logger.info(
            "STT ready — threshold=%.0f, silence=%.2fs, model=%s",
            self._volume_threshold, self._silence_duration, self._model_name,
        )

        try:
            while True:
                chunk: bytes = await mic_queue.get()
                rms = _rms(chunk)
                now = time.monotonic()

                if rms >= self._volume_threshold:
                    # ---- speech frame ----
                    if not in_speech:
                        # Rising edge: speech just started
                        in_speech = True
                        interrupt_sent = False
                        self._logger.debug("Speech start (rms=%.0f)", rms)

                    if in_speech and not interrupt_sent and llm_generating and llm_generating.is_set():
                        # Interrupt any active LLM/TTS response
                        await result_queue.put(INTERRUPT)
                        interrupt_sent = True
                        self._logger.info("INTERRUPT emitted.")

                    speech_buffer.append(chunk)
                    last_speech_time = now

                else:
                    # ---- silence frame ----
                    if in_speech:
                        speech_buffer.append(chunk)  # include trailing silence

                        elapsed_silence = now - last_speech_time
                        if elapsed_silence >= self._silence_duration:
                            # End of utterance
                            in_speech = False
                            pcm = b"".join(speech_buffer)
                            speech_buffer.clear()

                            duration = len(pcm) / (self._sample_rate * 2)  # int16 = 2 bytes
                            if duration >= self._min_speech_duration:
                                text = await asyncio.to_thread(self._transcribe, pcm)
                                if text:
                                    self._logger.info("Transcribed: %r", text)
                                    await result_queue.put(text)
                                else:
                                    self._logger.debug("Empty transcription, discarding.")
                            else:
                                self._logger.debug(
                                    "Speech too short (%.2fs < %.2fs), discarding.",
                                    duration, self._min_speech_duration,
                                )
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Transcription (runs in a thread pool thread)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the faster-whisper model on first use."""
        if self._model is not None:
            return
        self._logger.info("Loading Whisper model '%s' on %s...", self._model_name, self._device)
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        self._logger.info("Whisper model loaded.")

    def _transcribe(self, pcm: bytes) -> str:
        """Convert raw int16 PCM bytes to text using faster-whisper."""
        self._load_model()
        # Normalize int16 → float32 [-1, 1] as required by Whisper
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            samples,
            language="en",
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
