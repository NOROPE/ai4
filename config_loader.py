"""
config_loader.py — Profile-aware configuration loading.

Resolution order (later wins):
  1. Built-in defaults (hardcoded)
  2. config.json         — global fallback shared across all profiles
  3. profiles/<name>.json — profile-specific overrides
  4. Auto-derived paths  — based on profile name (overridable in profile JSON)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent

# All available Gemini voices (from planning/voices.txt)
VOICES: list[tuple[str, str]] = [
    ("Zephyr",        "Bright"),
    ("Puck",          "Upbeat"),
    ("Charon",        "Informative"),
    ("Kore",          "Firm"),
    ("Fenrir",        "Excitable"),
    ("Leda",          "Youthful"),
    ("Orus",          "Firm"),
    ("Aoede",         "Breezy"),
    ("Callirrhoe",    "Easy-going"),
    ("Autonoe",       "Bright"),
    ("Enceladus",     "Breathy"),
    ("Iapetus",       "Clear"),
    ("Umbriel",       "Easy-going"),
    ("Algieba",       "Smooth"),
    ("Despina",       "Smooth"),
    ("Erinome",       "Clear"),
    ("Algenib",       "Gravelly"),
    ("Rasalgethi",    "Informative"),
    ("Laomedeia",     "Upbeat"),
    ("Achernar",      "Soft"),
    ("Alnilam",       "Firm"),
    ("Schedar",       "Even"),
    ("Gacrux",        "Mature"),
    ("Pulcherrima",   "Forward"),
    ("Achird",        "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix",  "Gentle"),
    ("Sadachbia",     "Lively"),
    ("Sadaltager",    "Knowledgeable"),
    ("Sulafat",       "Warm"),
]

# Built-in defaults — lowest priority, never needs editing
_DEFAULTS: dict = {
    "model": "gemini-2.5-flash-native-audio-preview-12-2025",
    "system_instruction": "You are a helpful and friendly AI assistant.",
    "audio_buffer_seconds": 1.5,
    "buffer_clear_timeout_seconds": 2.0,
    "reconnect_interval_seconds": 2.0,
    "send_sample_rate": 16000,
    "receive_sample_rate": 24000,
    "chunk_size": 1024,
    "prevmsg_count": 30,
    "tool_mixins": [],
}


@dataclass
class ProfileConfig:
    """Fully resolved configuration for a single profile."""
    name: str = "default"

    # AI
    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    system_instruction: str = "You are a helpful and friendly AI assistant."

    # Audio
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    chunk_size: int = 1024
    audio_buffer_seconds: float = 1.5
    buffer_clear_timeout_seconds: float = 2.0

    # Connection
    reconnect_interval_seconds: float = 2.0

    # Paths (all absolute, profile-isolated)
    logs_dir: Path = Path(os.devnull)
    transcription_log_file: Path = Path(os.devnull)
    prevmsg_file: Path = Path(os.devnull)
    prevmsg_count: int = 30

    # PipeWire — each profile gets its own virtual sinks
    pipewire_sink_output: str = "AIOutput-default"
    pipewire_sink_input: str = "AIInput-default"

    # Docker — each profile gets its own container
    docker_container_name: str = "ai4-default"

    # Voice — None means prompt at startup
    voice: str | None = None

    # Tool mixins — list of mixin names to activate (e.g. ["system_info"])
    tool_mixins: list[str] = None  # type: ignore[assignment]

    # Behaviour flags
    teardown_sinks_on_exit: bool = True  # whether to remove virtual sinks on shutdown

    # Raw merged dict (for forward-compatible access to unknown keys)
    raw: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tool_mixins is None:
            self.tool_mixins = []
        if self.raw is None:
            self.raw = {}


def load_profile(name: str = "default") -> ProfileConfig:
    """
    Merge global config.json → profiles/<name>.json → auto-derived paths.
    Raises FileNotFoundError if the profile does not exist.
    """
    # 1. Built-in defaults
    merged: dict = dict(_DEFAULTS)

    # 2. Global config.json
    global_path = BASE_DIR / "config.json"
    if global_path.exists():
        with open(global_path) as f:
            merged.update(json.load(f))

    # 3. Profile-specific JSON
    profile_path = BASE_DIR / "profiles" / f"{name}.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile '{name}' not found. "
            f"Create profiles/{name}.json or choose from: {list_profiles()}"
        )
    with open(profile_path) as f:
        profile_data = json.load(f)
    merged.update(profile_data)

    # 4. Resolve paths — profile JSON can override any of these explicitly
    logs_dir = Path(merged.get("logs_dir", BASE_DIR / "logs" / name))
    transcription_log_file = Path(
        merged.get("transcription_log_file", logs_dir / "transcriptions.log")
    )
    prevmsg_file = Path(merged.get("prevmsg_file", BASE_DIR / "prevmsg" / f"{name}.txt"))

    # Ensure directories exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    prevmsg_file.parent.mkdir(parents=True, exist_ok=True)

    return ProfileConfig(
        name=name,
        model=merged.get("model", _DEFAULTS["model"]),
        system_instruction=merged.get("system_instruction", _DEFAULTS["system_instruction"]),
        send_sample_rate=int(merged.get("send_sample_rate", _DEFAULTS["send_sample_rate"])),
        receive_sample_rate=int(merged.get("receive_sample_rate", _DEFAULTS["receive_sample_rate"])),
        chunk_size=int(merged.get("chunk_size", _DEFAULTS["chunk_size"])),
        audio_buffer_seconds=float(merged.get("audio_buffer_seconds", _DEFAULTS["audio_buffer_seconds"])),
        buffer_clear_timeout_seconds=float(merged.get("buffer_clear_timeout_seconds", _DEFAULTS["buffer_clear_timeout_seconds"])),
        reconnect_interval_seconds=float(merged.get("reconnect_interval_seconds", _DEFAULTS["reconnect_interval_seconds"])),
        logs_dir=logs_dir,
        transcription_log_file=transcription_log_file,
        prevmsg_file=prevmsg_file,
        prevmsg_count=int(merged.get("prevmsg_count", _DEFAULTS["prevmsg_count"])),
        # PipeWire: default to profile-namespaced sinks so concurrent profiles don't clash
        pipewire_sink_output=merged.get("pipewire_sink_output", f"AIOutput-{name}"),
        pipewire_sink_input=merged.get("pipewire_sink_input", f"AIInput-{name}"),
        # Docker: default to a profile-namespaced container name
        docker_container_name=merged.get("docker_container_name", f"ai4-{name}"),
        teardown_sinks_on_exit=bool(merged.get("teardown_sinks_on_exit", True)),
        voice=merged.get("voice", None),  # None = prompt at startup
        tool_mixins=list(merged.get("tool_mixins", _DEFAULTS["tool_mixins"])),
        raw=merged,
    )


def list_profiles() -> list[str]:
    """Return the names of all available profiles."""
    profiles_dir = BASE_DIR / "profiles"
    if not profiles_dir.exists():
        return []
    return [p.stem for p in sorted(profiles_dir.glob("*.json"))]
