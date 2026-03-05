"""
PipeWire virtual audio sink management.
Creates/reuses AIOutput and AIInput virtual sinks and handles cleanup on shutdown.
"""

import asyncio
import logging
import subprocess

logger = logging.getLogger(__name__)

SINK_OUTPUT = "AIOutput"
SINK_INPUT = "AIInput"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _existing_node_names() -> set[str]:
    result = _run(["pw-cli", "list-objects", "Node"])
    names = set()
    for line in result.stdout.splitlines():
        if "node.name" in line:
            # e.g.  node.name = "AIOutput"
            parts = line.split("=", 1)
            if len(parts) == 2:
                names.add(parts[1].strip().strip('"'))
    return names


def create_virtual_sink(name: str) -> None:
    """Create a null-sink (virtual audio device) via pw-loopback / pactl."""
    existing = _existing_node_names()
    if name in existing:
        logger.info("PipeWire: sink '%s' already exists, reusing.", name)
        return

    # pactl works for both PulseAudio and PipeWire (pipewire-pulse)
    result = _run([
        "pactl", "load-module", "module-null-sink",
        f"sink_name={name}",
        f"sink_properties=device.description={name}",
    ])
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create virtual sink '{name}': {result.stderr.strip()}"
        )
    logger.info("PipeWire: created virtual sink '%s'.", name)


def remove_virtual_sink(name: str) -> None:
    """Unload the null-sink module for the given sink name."""
    # Find the module index
    result = _run(["pactl", "list", "modules", "short"])
    for line in result.stdout.splitlines():
        if "module-null-sink" in line and name in line:
            module_id = line.split()[0]
            _run(["pactl", "unload-module", module_id])
            logger.info("PipeWire: removed virtual sink '%s'.", name)
            return
    logger.debug("PipeWire: sink '%s' not found during cleanup.", name)


def setup_sinks(sink_output: str = SINK_OUTPUT, sink_input: str = SINK_INPUT) -> None:
    """Create both virtual sinks (idempotent)."""
    create_virtual_sink(sink_output)
    create_virtual_sink(sink_input)


def teardown_sinks(sink_output: str = SINK_OUTPUT, sink_input: str = SINK_INPUT) -> None:
    """Remove both virtual sinks."""
    remove_virtual_sink(sink_output)
    remove_virtual_sink(sink_input)


def set_default_sink(name: str) -> None:
    """Set the PulseAudio/PipeWire default output sink."""
    result = _run(["pactl", "set-default-sink", name])
    if result.returncode != 0:
        logger.warning("Could not set default sink to '%s': %s", name, result.stderr.strip())
    else:
        logger.info("PipeWire: default sink set to '%s'.", name)


def set_default_source(name: str) -> None:
    """Set the PulseAudio/PipeWire default input source."""
    result = _run(["pactl", "set-default-source", name])
    if result.returncode != 0:
        logger.warning("Could not set default source to '%s': %s", name, result.stderr.strip())
    else:
        logger.info("PipeWire: default source set to '%s'.", name)


def list_audio_devices(kind: str = "sinks") -> list[dict]:
    """Return a list of available sinks or sources via pactl."""
    result = _run(["pactl", "list", "short", kind])
    devices = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            devices.append({"index": parts[0], "name": parts[1]})
    return devices


def select_sink(prompt_label: str = "output", kind: str = "sinks") -> str | None:
    """
    Interactively prompt the user to select an audio sink or source,
    then set it as the PulseAudio default so sounddevice's 'pulse' device routes there.
    kind: 'sinks' for output devices, 'sources' for input/capture devices.
    Returns the selected device name, or None to use the system default.
    """
    devices = list_audio_devices(kind)
    if not devices:
        logger.warning("No audio %s found.", kind)
        return None

    label = "sinks" if kind == "sinks" else "sources (microphones)"
    print(f"\nAvailable audio {label} (for {prompt_label}):")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev['name']}")
    print(f"  [Enter] Use system default")

    try:
        choice = input(f"Select sink for {prompt_label}: ").strip()
        if choice == "":
            return None
        idx = int(choice)
        if 0 <= idx < len(devices):
            name = devices[idx]["name"]
            # Route the PulseAudio default so sounddevice's 'pulse' device uses it
            if kind == "sinks":
                set_default_sink(name)
            else:
                set_default_source(name)
            return name
        logger.warning("Invalid selection, using system default.")
        return None
    except (ValueError, EOFError):
        return None
