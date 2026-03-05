"""
audio/audio_io.py — Sounddevice I/O for Gemini Live API.
Handles mic capture → Gemini and Gemini audio → speakers,
including buffering, flush/discard sentinels, and transcription logging.
"""

import asyncio
import logging

import numpy as np
import sounddevice as sd

from audio.transcription import TranscriptionBuffer

logger = logging.getLogger(__name__)

# Sentinels for the playback queue
_FLUSH = object()    # end of turn: play remaining buffered audio
_DISCARD = object()  # interruption: drop buffered audio


def find_pulse_device() -> int | None:
    """
    Return the sounddevice index for the 'pulse' device, which routes audio
    through PulseAudio and honours the default sink/source set via pactl.
    """
    for i, dev in enumerate(sd.query_devices()):
        if dev["name"].lower() == "pulse":
            logger.debug("Using sounddevice 'pulse' (device #%d).", i)
            return i
    logger.warning("'pulse' sounddevice not found, falling back to system default.")
    return None


async def listen_and_send(
    session,
    input_device: int | None,
    send_sample_rate: int,
    chunk_size: int,
) -> None:
    """Capture mic audio and stream it to the Gemini session."""
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
    loop = asyncio.get_running_loop()

    def _mic_callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Mic status: %s", status)
        loop.call_soon_threadsafe(
            lambda d=bytes(indata): mic_queue.put_nowait(d) if not mic_queue.full() else None
        )

    with sd.RawInputStream(
        samplerate=send_sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=input_device,
        callback=_mic_callback,
    ):
        logger.info("Microphone open (device=%s).", input_device)
        while True:
            data = await mic_queue.get()
            await session.send_realtime_input(audio={"data": data, "mime_type": "audio/pcm"})


async def receive_and_play(
    session,
    output_device: int | None,
    receive_sample_rate: int,
    buffer_fill_bytes: int,
    buffer_clear_timeout: float,
    transcription_logger: logging.Logger,
) -> None:
    """
    Receive audio and transcriptions from Gemini.
    Buffers audio until buffer_fill_bytes is accumulated before playing.
    At end of turn, flushes remaining audio. After buffer_clear_timeout
    seconds of silence, discards stale buffered audio (handles interruptions).
    """
    play_queue: asyncio.Queue = asyncio.Queue()

    async def _player() -> None:
        buffer = bytearray()
        stream = sd.RawOutputStream(
            samplerate=receive_sample_rate,
            channels=1,
            dtype="int16",
            device=output_device,
            latency="high",  # larger internal buffer reduces underruns
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
            if buffer:
                await asyncio.to_thread(stream.write, bytes(buffer))
            stream.stop()
            stream.close()

    player_task = asyncio.create_task(_player())

    async def _clear_on_idle() -> None:
        while True:
            await asyncio.sleep(buffer_clear_timeout)
            if play_queue.empty():
                play_queue.put_nowait(_DISCARD)

    idle_task = asyncio.create_task(_clear_on_idle())

    user_buf = TranscriptionBuffer("User", transcription_logger)
    gemini_buf = TranscriptionBuffer("Gemini", transcription_logger)

    try:
        while True:
            turn = session.receive()
            async for response in turn:
                sc = response.server_content
                if sc:
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                play_queue.put_nowait(part.inline_data.data)
                            elif part.text:
                                logger.info("Gemini: %s", part.text)

                    if sc.input_transcription and sc.input_transcription.text:
                        user_buf.append(sc.input_transcription.text)

                    if sc.output_transcription and sc.output_transcription.text:
                        gemini_buf.append(sc.output_transcription.text)

            # Turn ended — flush audio tail then complete transcriptions
            play_queue.put_nowait(_FLUSH)
            user_buf.flush()
            gemini_buf.flush()
    finally:
        idle_task.cancel()
        player_task.cancel()
