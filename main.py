"""
main.py — AI4 Assistant
Phases implemented: 1 (skeleton), 2 (logging), 3 (PipeWire virtual sinks),
                   5 (Gemini Live API with audio I/O)
"""

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from human_ai.stt import STT, INTERRUPT
from human_ai.llm import LLM
from human_ai.tts import TTS

from ai_io import find_device, mic_capture, audio_player, _DISCARD
from audio.pipewire import setup_sinks, teardown_sinks
from config_loader import ProfileConfig, VOICES, list_profiles, load_profile
from tools import Tools


# ---------------------------------------------------------------------------
# Custom log level for transcriptions
# ---------------------------------------------------------------------------
TRANSCRIPTION_LEVEL = 25
logging.addLevelName(TRANSCRIPTION_LEVEL, "TRANSCRIPTION")


class TranscriptionOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == TRANSCRIPTION_LEVEL


# ---------------------------------------------------------------------------
# Voice selection — interactive prompt at startup
# ---------------------------------------------------------------------------


def select_voice(default: str | None = None) -> str:
    """Prompt the user to pick a voice. Returns the voice name."""
    print("\nAvailable voices:")
    col_width = max(len(name) for name, _ in VOICES) + 2
    for i, (name, style) in enumerate(VOICES):
        print(f"  [{i:2d}] {name:<{col_width}} {style}")
    default_name = default or VOICES[0][0]
    print(f"  [Enter] Use '{default_name}'")

    try:
        choice = input("Select voice: ").strip()
        if choice == "":
            return default_name
        idx = int(choice)
        if 0 <= idx < len(VOICES):
            return VOICES[idx][0]
        print("Invalid selection, using default.")
        return default_name
    except (ValueError, EOFError):
        return default_name


# ---------------------------------------------------------------------------
# Profile selection — interactive prompt at startup
# ---------------------------------------------------------------------------


def select_profile() -> str:
    """Prompt the user to choose a profile from the profiles/ directory."""
    profiles = list_profiles()
    if not profiles:
        print("No profiles found in profiles/. Using 'default'.")
        return "default"

    if len(profiles) == 1:
        print(f"Using profile: {profiles[0]}")
        return profiles[0]

    print("\nAvailable profiles:")
    for i, name in enumerate(profiles):
        print(f"  [{i}] {name}")
    print(f"  [Enter] Use '{profiles[0]}'")

    try:
        choice = input("Select profile: ").strip()
        if choice == "":
            return profiles[0]
        idx = int(choice)
        if 0 <= idx < len(profiles):
            return profiles[idx]
        print("Invalid selection, using first profile.")
        return profiles[0]
    except (ValueError, EOFError):
        return profiles[0]


# ---------------------------------------------------------------------------
# Logging — configured per-profile
# ---------------------------------------------------------------------------


def setup_logging(cfg: ProfileConfig) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    # Console — all messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(console_handler)

    # File — transcriptions only, no timestamps/metadata
    cfg.transcription_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(cfg.transcription_log_file, encoding="utf-8")
    file_handler.setLevel(TRANSCRIPTION_LEVEL)
    file_handler.addFilter(TranscriptionOnlyFilter())
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # Suppress noisy third-party debug output
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    return logging.getLogger(f"ai4.{cfg.name}")


# ---------------------------------------------------------------------------
# PipeWire audio setup — uses profile's sink names
# ---------------------------------------------------------------------------


def setup_audio(cfg: ProfileConfig, logger: logging.Logger) -> None:
    logger.info("Setting up PipeWire virtual sinks for profile '%s'...", cfg.name)
    try:
        setup_sinks(
            sink_output=cfg.pipewire_sink_output,
            sink_input=cfg.pipewire_sink_input,
        )
        logger.info("Virtual sinks created. Use pavucontrol to route streams.")
    except RuntimeError as e:
        logger.error("PipeWire setup failed: %s", e)


def teardown_audio(cfg: ProfileConfig, logger: logging.Logger) -> None:
    if not cfg.teardown_sinks_on_exit:
        logger.info("Skipping PipeWire sink teardown (teardown_sinks_on_exit=false).")
        return
    logger.info("Removing PipeWire virtual sinks for profile '%s'...", cfg.name)
    try:
        teardown_sinks(
            sink_output=cfg.pipewire_sink_output,
            sink_input=cfg.pipewire_sink_input,
        )
    except Exception as e:
        logger.warning("PipeWire teardown error: %s", e)


# ---------------------------------------------------------------------------
# AI Live session
# ---------------------------------------------------------------------------


async def run_session(
    cfg: ProfileConfig,
    stop_event: asyncio.Event,
    logger: logging.Logger,
    tools: Tools,
) -> None:
    """Connect the STT → LLM → TTS pipeline and run until stop_event is set."""
    # Discover audio devices
    input_device = find_device(cfg.pipewire_sink_input, kind="input")
    output_device = find_device(cfg.pipewire_sink_output, kind="output")

    buffer_fill_bytes = int(cfg.receive_sample_rate * 2 * cfg.audio_buffer_seconds)

    # Queues
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)
    stt_result_queue: asyncio.Queue = asyncio.Queue()
    audio_out_queue: asyncio.Queue = asyncio.Queue()

    # Components
    base_url: str = cfg.raw.get("base_url", "http://localhost:11434/v1")
    llm = LLM(base_url=base_url, config=cfg, tools=tools)
    await llm.setup()  # also calls tools.setup()
    stt = STT(config=cfg)
    tts = TTS(config=cfg)

    # Event set while the LLM/TTS is generating — STT uses it to emit INTERRUPT.
    llm_generating = asyncio.Event()

    async def _llm_tts_turn(text: str) -> None:
        """Run one LLM generate + TTS pass, bracketed by the generating flag."""
        llm_generating.set()
        try:
            token_queue: asyncio.Queue = asyncio.Queue()
            llm.generate(text, token_queue)
            await tts.run(token_queue, audio_out_queue)
        finally:
            llm_generating.clear()

    # Background I/O tasks
    mic_task = asyncio.create_task(
        mic_capture(mic_queue, input_device, cfg.send_sample_rate, cfg.chunk_size, stop_event),
        name="mic_capture",
    )
    player_task = asyncio.create_task(
        audio_player(
            audio_out_queue, output_device,
            cfg.receive_sample_rate, buffer_fill_bytes,
            cfg.buffer_clear_timeout_seconds,
        ),
        name="audio_player",
    )
    stt_task = asyncio.create_task(
        stt.run(mic_queue, stt_result_queue, llm_generating),
        name="stt",
    )

    logger.info("Session started. Listening...")
    silence_ping_interval: float = float(cfg.raw.get("silence_ping_interval", 5.0))
    last_activity = asyncio.get_event_loop().time()
    try:
        while not stop_event.is_set():
            try:
                result = await asyncio.wait_for(stt_result_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                # No speech — ping the LLM with "..." if silent long enough and idle
                if (
                    not llm_generating.is_set()
                    and asyncio.get_event_loop().time() - last_activity >= silence_ping_interval
                ):
                    last_activity = asyncio.get_event_loop().time()
                    logger.debug("Silence ping → LLM")
                    asyncio.create_task(_llm_tts_turn("..."), name="llm_tts_turn")
                continue

            last_activity = asyncio.get_event_loop().time()

            if result is INTERRUPT:
                # User spoke mid-response — stop the LLM and drop buffered audio
                logger.info("STT interrupt: stopping LLM and discarding audio buffer.")
                llm.interrupt()
                audio_out_queue.put_nowait(_DISCARD)
                continue

            if not isinstance(result, str) or not result.strip():
                continue

            logger.info("STT → LLM: %r", result)
            asyncio.create_task(_llm_tts_turn(result), name="llm_tts_turn")
    finally:
        mic_task.cancel()
        stt_task.cancel()
        player_task.cancel()
        await llm.shutdown()  # also calls tools.teardown()
        await asyncio.gather(mic_task, stt_task, player_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    profile_name = select_profile()

    try:
        cfg = load_profile(profile_name)
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] {e}")

    # Voice: use profile value if set, otherwise prompt
    if cfg.voice is None:
        cfg.voice = select_voice()
    else:
        print(f"Using voice: {cfg.voice}")

    logger = setup_logging(cfg)
    logger.info("AI4 starting up with profile '%s'.", cfg.name)
    logger.info("Transcription log: %s", cfg.transcription_log_file)

    setup_audio(cfg, logger)

    # Tool system — instantiate mixins from profile (setup called inside run_session via LLM)
    tools = Tools(cfg.tool_mixins, config=cfg)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("Received signal %s, shutting down...", sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    logger.log(TRANSCRIPTION_LEVEL, "[system] Session started.")

    await run_session(cfg, stop_event, logger, tools)

    teardown_audio(cfg, logger)
    logger.info("AI4 shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
