"""
live2d_cli.py — Interactive CLI to test Live2D / VTube Studio requests.
Run with: python live2d_cli.py
"""

import asyncio
import logging
from tools.mixins.live2d_mixin import Live2DMixin

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live2d_cli")


class _MockConfig:
    name = "cli"
    raw = {
        "vtube_studio_websockets_url": "ws://localhost:8001",
        "live2d_icon_path": "media/live2dpluginicon.png",
        # "vtube_studio_reconnect": True,       # set to False to disable the reconnect loop
        # "vtube_studio_reconnect_interval": 1.0, # seconds between health-check polls
    }


COMMANDS = {
    "status":("Get model status",      []),
    "pos":  ("Set position",          ["x (-1000 to 1000, edges ≈ ±1)", "y (-1000 to 1000, edges ≈ ±1)", "time (0-2s) [0.5]"]),
    "rot":  ("Set rotation",          ["rotation (-360 to 360, + = CW)", "time (0-2s) [0.5]"]),
    "size": ("Set size",              ["size (-100 to 100)", "time (0-2s) [0.5]"]),
    "rpos": ("Move relative",         ["dx", "dy", "time (0-2s) [0.5]"]),
    "rrot": ("Rotate relative",       ["rotation offset (+ = CW)", "time (0-2s) [0.5]"]),
    "rsize":("Resize relative",       ["size offset", "time (0-2s) [0.5]"]),
}


async def cli():
    mixin = Live2DMixin(config=_MockConfig())
    await mixin.setup()
    print("\nConnected and authenticated. Type 'help' for commands, 'q' to quit.\n")

    def arg(parts, index, prompt, default=None):
        """Return parts[index] if present, otherwise prompt the user."""
        if index < len(parts):
            return parts[index]
        val = input(f"  {prompt}: ").strip()
        return val if val else (str(default) if default is not None else val)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        parts = line.split()
        if not parts:
            continue
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("help", "h", "?"):
            print("\nCommands:")
            for key, (desc, params) in COMMANDS.items():
                print(f"  {key:<6} — {desc}  ({', '.join(params)})" if params else f"  {key:<6} — {desc}")
            print()
        elif cmd == "status":
            result = await mixin.get_model_status()
            print(f"  → {result}")
        elif cmd == "pos":
            try:
                x = float(arg(args, 0, "x"))
                y = float(arg(args, 1, "y"))
                t = arg(args, 2, "time [0.5]", 0.5)
                result = await mixin.set_model_position(x, y, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif cmd == "rot":
            try:
                r = float(arg(args, 0, "rotation"))
                t = arg(args, 1, "time [0.5]", 0.5)
                result = await mixin.set_model_rotation(r, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif cmd == "size":
            try:
                s = float(arg(args, 0, "size"))
                t = arg(args, 1, "time [0.5]", 0.5)
                result = await mixin.set_model_size(s, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif cmd == "rpos":
            try:
                dx = float(arg(args, 0, "dx"))
                dy = float(arg(args, 1, "dy"))
                t = arg(args, 2, "time [0.5]", 0.5)
                result = await mixin.move_model_relative(dx, dy, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif cmd == "rrot":
            try:
                r = float(arg(args, 0, "rotation offset"))
                t = arg(args, 1, "time [0.5]", 0.5)
                result = await mixin.rotate_model_relative(r, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif cmd == "rsize":
            try:
                s = float(arg(args, 0, "size offset"))
                t = arg(args, 1, "time [0.5]", 0.5)
                result = await mixin.resize_model_relative(s, float(t))
                print(f"  → {result}")
            except ValueError as e:
                print(f"  Invalid input: {e}")
        else:
            print(f"  Unknown command '{cmd}'. Type 'help' for commands.")

    await mixin.teardown()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(cli())
