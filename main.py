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

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai_io import listen_and_send, receive_and_play
from audio.pipewire import setup_sinks, teardown_sinks
from config_loader import ProfileConfig, VOICES, list_profiles, load_profile
from tools import Tools

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit("[ERROR] No Google API key found. Set GEMINI_API_KEY in .env")

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
# Gemini Live session
# ---------------------------------------------------------------------------


async def run_session(
    cfg: ProfileConfig,
    stop_event: asyncio.Event,
    logger: logging.Logger,
    tools: Tools,
) -> None:
    """Connect to Gemini Live API and run audio I/O; reconnect on failure."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    buffer_fill_bytes = int(cfg.receive_sample_rate * 2 * cfg.audio_buffer_seconds)

    # Load previous conversation context for this profile
    prev_context = ""
    if cfg.prevmsg_file.exists():
        prev_context = cfg.prevmsg_file.read_text(encoding="utf-8").strip()

    system_instruction = cfg.system_instruction
    if prev_context:
        system_instruction += f"\n\nPrevious conversation context:\n{prev_context}"

    live_config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=cfg.voice,
                )
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[tools.get_tool_config()] if tools.has_tools else [],
    )

    while not stop_event.is_set():
        try:
            logger.info("Connecting to Gemini Live API (model=%s)...", cfg.model)
            async with client.aio.live.connect(model=cfg.model, config=live_config) as session:
                logger.info("Connected. Start speaking!")
                listen_task = asyncio.create_task(
                    listen_and_send(session, None, cfg.send_sample_rate, cfg.chunk_size)
                )
                play_task = asyncio.create_task(
                    receive_and_play(
                        session, None,
                        cfg.receive_sample_rate, buffer_fill_bytes,
                        cfg.buffer_clear_timeout_seconds, logger,
                        tools,
                    )
                )
                done, pending = await asyncio.wait(
                    [listen_task, play_task, asyncio.create_task(stop_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                for t in done:
                    if not t.cancelled() and t.exception():
                        raise t.exception()  # type: ignore[misc]

        except asyncio.CancelledError:
            break
        except Exception as exc:
            if stop_event.is_set():
                break
            logger.error(
                "Session error: %s — reconnecting in %.1fs", exc, cfg.reconnect_interval_seconds
            )
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=cfg.reconnect_interval_seconds
                )
            except asyncio.TimeoutError:
                pass


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
    logger.info("API key present: %s", bool(GEMINI_API_KEY))

    setup_audio(cfg, logger)

    # Tool system — instantiate and set up mixins from profile
    tools = Tools(cfg.tool_mixins, config=cfg)
    await tools.setup()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("Received signal %s, shutting down...", sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    logger.log(TRANSCRIPTION_LEVEL, "[system] Session started.")

    await run_session(cfg, stop_event, logger, tools)

    await tools.teardown()
    teardown_audio(cfg, logger)
    logger.info("AI4 shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
