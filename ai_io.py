"""
ai_io.py — Sounddevice I/O helpers for the STT → LLM → TTS pipeline.

Provides:
  mic_capture   — streams raw PCM from the microphone into an asyncio.Queue
  audio_player  — reads PCM chunks (plus _FLUSH / _DISCARD sentinels) and
                  plays them through a sounddevice output stream
  find_device   — resolves a PipeWire/PulseAudio device name to a device index
"""

import asyncio
import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Sentinels for the audio_player queue
_FLUSH = object()    # end of turn: play remaining buffered audio
_DISCARD = object()  # interruption: drop buffered audio


def find_device(name: str | None = None, kind: str = "output") -> int | None:
    """
    Return the sounddevice index for the named PulseAudio/PipeWire device.
    kind: 'output' (max_output_channels > 0) or 'input' (max_input_channels > 0).
    If name is given, searches for a device whose name contains it (case-insensitive).
    Falls back to the 'pulse' passthrough device, then system default.
    """
    ch_key = "max_output_channels" if kind == "output" else "max_input_channels"
    devices = list(enumerate(sd.query_devices()))
    if name:
        for i, dev in devices:
            if dev.get(ch_key, 0) > 0 and name.lower() in dev["name"].lower():
                logger.debug("Using sounddevice '%s' (device #%d) for %s sink '%s'.", dev["name"], i, kind, name)
                return i
        logger.warning("Sounddevice for %s sink '%s' not found, falling back to 'pulse'.", kind, name)
    for i, dev in devices:
        if dev.get(ch_key, 0) > 0 and dev["name"].lower() == "pulse":
            logger.debug("Using sounddevice 'pulse' (device #%d) for %s.", i, kind)
            return i
    logger.warning("'pulse' sounddevice not found for %s, falling back to system default.", kind)
    return None


async def mic_capture(
    mic_queue: asyncio.Queue,
    input_device: int | None,
    sample_rate: int,
    chunk_size: int,
    stop_event: asyncio.Event | None = None,
) -> None:
    """
    Capture microphone audio and push raw PCM bytes into *mic_queue*.
    Runs until *stop_event* is set (or forever if None).
    """
    loop = asyncio.get_running_loop()

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Mic status: %s", status)
        data = bytes(indata)
        loop.call_soon_threadsafe(
            lambda d=data: None if mic_queue.full() else mic_queue.put_nowait(d)
        )

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=input_device,
        callback=_callback,
    ):
        logger.info("Microphone open (device=%s).", input_device)
        if stop_event is not None:
            await stop_event.wait()
        else:
            await asyncio.get_event_loop().create_future()  # run forever


async def audio_player(
    play_queue: asyncio.Queue,
    output_device: int | None,
    sample_rate: int,
    buffer_fill_bytes: int,
    buffer_clear_timeout: float,
) -> None:
    """
    Play audio chunks from *play_queue* through a sounddevice output stream.

    Sentinels:
      _FLUSH   — play any buffered audio immediately (end of TTS turn)
      _DISCARD — drop buffered audio without playing (interruption)
    """
    async def _idle_discard() -> None:
        """After buffer_clear_timeout seconds of queue silence, discard stale audio."""
        while True:
            await asyncio.sleep(buffer_clear_timeout)
            if play_queue.empty():
                play_queue.put_nowait(_DISCARD)

    idle_task = asyncio.create_task(_idle_discard())
    buffer = bytearray()
    stream = sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=output_device,
        latency="high",
    )
    stream.start()
    try:
        while True:
            chunk = await play_queue.get()
            if chunk is _DISCARD:
                buffer.clear()
                continue
            if chunk is _FLUSH:
                if buffer:
                    await asyncio.to_thread(stream.write, bytes(buffer))
                    buffer.clear()
                continue
            buffer.extend(chunk)
            if len(buffer) >= buffer_fill_bytes:
                await asyncio.to_thread(stream.write, bytes(buffer))
                buffer.clear()
    finally:
        idle_task.cancel()
        if buffer:
            await asyncio.to_thread(stream.write, bytes(buffer))
        stream.stop()
        stream.close()
