"""
main.py — AI4 Assistant
Phases implemented: 1 (skeleton), 2 (logging), 3 (PipeWire virtual sinks),
                   5 (Gemini Live API with audio I/O)
"""

import asyncio
import logging
import logging.handlers
import signal
import sys

from human_ai.pipeline import Pipeline

from audio.pipewire import setup_sinks, teardown_sinks
from config_loader import ProfileConfig, list_profiles, load_profile
from tools import Tools




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

    # Suppress noisy third-party debug output
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

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
    pipeline = Pipeline(cfg, tools, logger)
    await pipeline.run(stop_event)


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


    await run_session(cfg, stop_event, logger, tools)

    teardown_audio(cfg, logger)
    logger.info("AI4 shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
