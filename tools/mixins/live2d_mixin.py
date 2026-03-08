"""
Mixin for Live2D
uses the VTube Studio WebSocket API to control Live2D models in VTube Studio.
Requires VTube Studio v1.8.0+ and the WebSocket API plugin.

Docs:
https://github.com/DenchiSoft/VTubeStudio

"""

from __future__ import annotations
from typing import Any
from pathlib import Path
import websockets
from websockets.asyncio.client import ClientConnection
import base64
import json
import asyncio
try:
    from pynput import mouse as pynput_mouse
except ImportError:
    pynput_mouse = None
from tools.base_mixin import ToolMixin, tool_function, fire_and_forget

class Live2DMixin(ToolMixin):
    """Provides tools for controlling Live2D models in VTube Studio."""

    MIXIN_NAME = "live2d"

    def __init__(self, config=None, **kwargs) -> None: # no vol needed, lipsync will be handled by VTube Studio
        super().__init__(config, **kwargs)
        self.websocket_url = self.config.raw.get("vtube_studio_websockets_url", "ws://localhost:8001")
        self.websocket_connection: ClientConnection | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._mouse_task: asyncio.Task | None = None
        self._ws_lock = asyncio.Lock()
        self._reconnect_enabled = bool(self.config.raw.get("vtube_studio_reconnect", True))
        self._reconnect_interval = float(self.config.raw.get("vtube_studio_reconnect_interval", 1.0))
        self._mouse_tracking_enabled = bool(self.config.raw.get("vtube_studio_mouse_tracking", False))
        self._mouse_tracking_fps = float(self.config.raw.get("vtube_studio_mouse_tracking_fps", 30.0))
        self._mouse_tracking_active = self._mouse_tracking_enabled  # runtime toggle
        self.icon_path = self.config.raw.get("live2d_icon_path", "")

        _token_dir = Path(__file__).parent / "data" / self.config.name
        _token_dir.mkdir(parents=True, exist_ok=True)
        self._token_file = _token_dir / "vtube_token.txt"

        with open(self.icon_path, "rb") as f:
            self.icon = base64.b64encode(f.read()).decode("utf-8")

        


    async def _send_recv(self, request: str) -> dict:
        """Send a request and receive a response, serialized via lock.
        On any websocket error, nulls the connection (triggering the reconnect loop) and raises ConnectionError.
        """
        if self.websocket_connection is None:
            raise ConnectionError("Not connected to VTube Studio.")
        async with self._ws_lock:
            try:
                await self.websocket_connection.send(request)
                return json.loads(await self.websocket_connection.recv())
            except Exception as e:
                self.websocket_connection = None  # signal disconnected so reconnect loop fires
                raise ConnectionError(f"WebSocket error: {e}") from e

    def _load_token(self) -> str | None:
        """Load the stored auth token from disk, if it exists."""
        if self._token_file.exists():
            token = self._token_file.read_text().strip()
            return token if token else None
        return None

    def _save_token(self, token: str) -> None:
        """Persist the auth token to disk for reuse across sessions."""
        self._token_file.write_text(token)

    async def _request_new_token(self) -> str:
        """Request a new auth token from VTube Studio (shows approval popup to user).
        Waits until the user clicks Allow or Deny in VTube Studio.
        """
        token_request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"{self.config.name} - AuthTokenRequest",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": f"AI plugin for VTubeStudio - {self.config.name}",
                "pluginDeveloper": "Winston King",
                "pluginIcon": self.icon
            }
        }
        response = await self._send_recv(json.dumps(token_request))
        if response.get("messageType") == "APIError":
            error_msg = response.get("data", {}).get("message", "Unknown error")
            raise PermissionError(f"User denied API access: {error_msg}")
        token = response.get("data", {}).get("authenticationToken", "")
        if not token:
            raise ValueError(f"No token in AuthenticationTokenResponse: {response}")
        return token

    async def _authenticate_with_token(self, token: str) -> bool:
        """Attempt to authenticate for this session using an existing token.
        Returns True if authenticated, False if the token is invalid/revoked.
        """
        auth_request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"{self.config.name} - AuthRequest",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": f"AI plugin for VTubeStudio - {self.config.name}",
                "pluginDeveloper": "Winston King",
                "authenticationToken": token
            }
        }
        response = await self._send_recv(json.dumps(auth_request))
        return response.get("data", {}).get("authenticated", False)

    async def authenticate(self) -> None:
        """Authenticate with VTube Studio.

        Reuses a stored token if available so the user is only prompted once.
        If the stored token is invalid or revoked, requests a new one.
        """
        if not self.websocket_connection:
            raise ConnectionError("WebSocket connection not established. Cannot authenticate.")

        # Try reusing the stored token first (no popup needed)
        token = self._load_token()
        if token:
            self.logger.info("Found stored auth token, attempting session authentication...")
            if await self._authenticate_with_token(token):
                self.logger.info("Authenticated using stored token.")
                return
            self.logger.warning("Stored token is invalid or revoked. Requesting a new one...")

        # Request a new token — user must approve the popup in VTube Studio
        self.logger.info("Requesting new auth token (approve the popup in VTube Studio)...")
        token = await self._request_new_token()
        self._save_token(token)
        self.logger.info("New token received and saved.")

        if not await self._authenticate_with_token(token):
            raise PermissionError("Authentication failed immediately after receiving new token.")

    

    async def setup(self):
        """Establish WebSocket connection to VTube Studio."""
        try:
            self.websocket_connection = await websockets.connect(self.websocket_url, ping_interval=None)
            self.logger.info(f"Connected to VTube Studio at {self.websocket_url}")
            status_response = await self._send_recv(
                json.dumps({
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": f"{self.config.name} - StatusRequest",
                    "messageType": "APIStateRequest"
                })
            )
            status = json.dumps(status_response, indent=2)
            self.logger.info(f"API status: \n{status}")
            try:
                await self.authenticate()
                self.logger.info("Authenticated with VTube Studio.")
            except Exception as e:
                self.logger.error(f"Failed to authenticate with VTube Studio: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                await self.setup()
                return
        except Exception as e:
            self.logger.error(f"Failed to connect to VTube Studio: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
            await self.setup()
            return

        # Cancel any previous reconnect loop before starting a fresh one
        if self._reconnect_enabled:
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

        # Start mouse tracking if enabled
        if self._mouse_tracking_enabled:
            if self._mouse_task and not self._mouse_task.done():
                self._mouse_task.cancel()
            self._mouse_tracking_active = True
            self._mouse_task = asyncio.create_task(self._mouse_tracking_loop())
            self.logger.info("Mouse tracking started.")

    async def _mouse_tracking_loop(self) -> None:
        """Reads mouse position and injects it into VTube Studio eye parameters at _mouse_tracking_fps.

        Backends (auto-detected at loop start):
        - AI4 GNOME Shell extension (Wayland): queries org.ai4.CursorPosition.GetPosition via gdbus.
          Works across all windows including native Wayland clients (browsers, etc.).
          Install: copy gnome-cursor-ext/ to ~/.local/share/gnome-shell/extensions/cursor-position@ai4/
        - pynput (X11 fallback): polls Controller.position.  Works on X11 but
          freezes when the cursor enters a native Wayland window.
        """
        import subprocess, re, shutil

        # --- Detect cursor-position backend ---
        use_gnome_ext = False
        _gnome_pos_args = [
            "gdbus", "call", "--session",
            "--dest", "org.ai4.CursorPosition",
            "--object-path", "/org/ai4/CursorPosition",
            "--method", "org.ai4.CursorPosition.GetPosition",
        ]
        _gnome_size_args = [
            "gdbus", "call", "--session",
            "--dest", "org.ai4.CursorPosition",
            "--object-path", "/org/ai4/CursorPosition",
            "--method", "org.ai4.CursorPosition.GetScreenSize",
        ]

        if shutil.which("gdbus"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *_gnome_pos_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    use_gnome_ext = True
                    self.logger.info("Mouse tracking: using AI4 GNOME Shell extension (Wayland-safe).")
                else:
                    self.logger.warning(
                        "AI4 GNOME cursor extension not available (rc=%d, out=%s). "
                        "Install it: cp gnome-cursor-ext/* ~/.local/share/gnome-shell/extensions/cursor-position@ai4/ "
                        "&& gnome-extensions enable cursor-position@ai4 (then re-login). Falling back to pynput.",
                        proc.returncode, stdout.decode().strip(),
                    )
            except Exception as e:
                self.logger.warning("GNOME extension DBus probe exception: %s. Falling back to pynput.", e)
        else:
            self.logger.info("gdbus not found on PATH. Falling back to pynput.")

        if not use_gnome_ext:
            if pynput_mouse is None:
                self.logger.error(
                    "Mouse tracking unavailable: pynput not installed and GNOME extension not accessible."
                )
                return
            self.logger.info("Mouse tracking: using pynput backend (X11).")

        # --- Detect screen size ---
        sw = int(self.config.raw.get("screen_width", 0))
        sh = int(self.config.raw.get("screen_height", 0))

        if (not sw or not sh) and use_gnome_ext:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *_gnome_size_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    m = re.search(r"\((\d+),\s*(\d+)\)", stdout.decode())
                    if m:
                        sw, sh = int(m.group(1)), int(m.group(2))
            except Exception:
                pass

        if not sw or not sh:
            try:
                out = subprocess.check_output(["xrandr"], text=True)
                m = re.search(r"current (\d+) x (\d+)", out)
                if m:
                    sw, sh = int(m.group(1)), int(m.group(2))
            except Exception:
                pass
        if not sw or not sh:
            sw, sh = 1920, 1080
            self.logger.warning(
                "Could not detect screen size, defaulting to %dx%d. "
                "Set screen_width/screen_height in config to override.", sw, sh,
            )
        else:
            self.logger.info("Mouse tracking screen size: %dx%d", sw, sh)

        interval = 1.0 / self._mouse_tracking_fps
        _pos_re = re.compile(r"\((-?\d+),\s*(-?\d+)\)")
        mouse_ctrl: Any = pynput_mouse.Controller() if not use_gnome_ext and pynput_mouse else None

        try:
            while True:
                mx, my = None, None

                if use_gnome_ext:
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *_gnome_pos_args,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, _ = await proc.communicate()
                        if proc.returncode == 0:
                            m = _pos_re.search(stdout.decode())
                            if m:
                                mx, my = int(m.group(1)), int(m.group(2))
                    except Exception:
                        pass
                else:
                    try:
                        mx, my = mouse_ctrl.position
                    except Exception:
                        pass

                if mx is None or my is None:
                    await asyncio.sleep(interval)
                    continue

                x_norm = max(-1.0, min(1.0, (mx / sw) * 2 - 1))
                y_norm = max(-1.0, min(1.0, -((my / sh) * 2 - 1)))  # flip Y: top of screen = positive

                request = self.create_request(
                    message_type="InjectParameterDataRequest",
                    data={
                        "faceFound": False,
                        "mode": "set",
                        "parameterValues": [
                            {"id": "EyeRightX", "value": x_norm},
                            {"id": "EyeRightY", "value": y_norm},
                            {"id": "EyeLeftX",  "value": x_norm},
                            {"id": "EyeLeftY",  "value": y_norm},
                        ]
                    }
                )
                if self._mouse_tracking_active:
                    try:
                        await self._send_recv(request)
                    except ConnectionError:
                        pass  # reconnect loop handles recovery
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def _reconnect_loop(self):
        """Waits for the websocket to close, then spawns a new setup(). No polling — zero extra traffic."""
        ws = self.websocket_connection
        if ws is None:
            return
        try:
            await ws.wait_closed()
        except asyncio.CancelledError:
            return
        except Exception:
            pass
        if self.websocket_connection is ws:  # only act if setup() hasn't already replaced it
            self.websocket_connection = None
        self.logger.warning("VTube Studio connection closed. Reconnecting...")
        asyncio.create_task(self.setup())



    async def teardown(self):
        """Close WebSocket connection."""
        if self._mouse_task and not self._mouse_task.done():
            self._mouse_task.cancel()
            try:
                await self._mouse_task
            except asyncio.CancelledError:
                pass
            self._mouse_task = None
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        if self.websocket_connection:
            await self.websocket_connection.close()
            self.logger.info("WebSocket connection to VTube Studio closed.")

    def create_request(self, message_type: str, description="", **kwargs) -> str:
        """function that makes a request to the VTube Studio WebSocket API with the given description and additional parameters.
            Extra parameters are passed as keyword arguments and included in the request body.
        """
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"{self.config.name} - {description if description else message_type}",
            "messageType": message_type
        }
        request.update(kwargs)
        return json.dumps(request)

    # tool functions

    @tool_function(
        description="Get the current position, rotation, size, and name of the Live2D model.",
        parameter_descriptions={},
    )
    async def get_model_status(self) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(message_type="CurrentModelRequest")
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"
        data = response.get("data", {})
        pos = data.get("modelPosition", {})
        x        = pos.get("positionX", "N/A")
        y        = pos.get("positionY", "N/A")
        rotation = pos.get("rotation",  "N/A")
        size     = pos.get("size",      "N/A")
        name     = data.get("modelName", "unknown")
        return f"Model: {name} | position: ({x}, {y}) | rotation: {rotation}° | size: {size}"

    @tool_function(
        description=(
            "Move, rotate, and/or resize the avatar by relative offsets from its current pose. "
            "Pass only the values you want to change; leave others at 0. "
            "Use this to spin, slide, bounce, or resize the avatar as an emote."
        ),
        parameter_descriptions={
            "dx": "Horizontal offset (-1 to 1, positive = right)",
            "dy": "Vertical offset (-1 to 1, positive = up)",
            "drotation": "Rotation offset in degrees (positive = clockwise)",
            "dsize": "Size offset (-100 to 100, positive = bigger)",
            "time_in_seconds": "Animation duration in seconds (0 to 2)",
        },
    )
    @fire_and_forget
    async def move_model(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        drotation: float = 0.0,
        dsize: float = 0.0,
        time_in_seconds: float = 0.5,
    ) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(
                message_type="MoveModelRequest",
                data={
                    "timeInSeconds": time_in_seconds,
                    "valuesAreRelativeToModel": True,
                    "positionX": dx,
                    "positionY": dy,
                    "rotation": drotation,
                    "size": dsize,
                    "behavior": "NON_BLOCKING",
                },
            )
            await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        return f"Model adjusted (dx={dx}, dy={dy}, drotation={drotation}, dsize={dsize})."

    @tool_function(
        description="List all available expressions for the current Live2D model.",
        parameter_descriptions={},
    )
    async def list_expressions(self) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(message_type="ExpressionStateRequest", data={"details": True})
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"
        expressions = response.get("data", {}).get("expressions", [])
        if not expressions:
            return "No expressions found. Create them in VTube Studio under the Expressions tab."
        lines = [f"- {e['file']} (active: {e.get('active', False)})" for e in expressions]
        return "Expressions:\n" + "\n".join(lines)

    @tool_function(
        description=(
            "Activate or deactivate a facial expression on the avatar. "
            "Use list_expressions to see available names. "
            "When activating, the expression auto-deactivates after `timeout` seconds."
        ),
        parameter_descriptions={
            "expression_file": "Expression filename, e.g. 'happy.exp3.json'",
            "active": "True to activate, False to deactivate",
            "timeout": "Seconds before auto-deactivating (only used when active=True, default 20)",
        },
    )
    @fire_and_forget
    async def set_expression(
        self,
        expression_file: str,
        active: bool = True,
        timeout: float = 20.0,
    ) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(
                message_type="ExpressionActivationRequest",
                data={"expressionFile": expression_file, "active": active},
            )
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"

        if active and timeout > 0:
            async def _auto_deactivate() -> None:
                await asyncio.sleep(timeout)
                try:
                    req = self.create_request(
                        message_type="ExpressionActivationRequest",
                        data={"expressionFile": expression_file, "active": False},
                    )
                    await self._send_recv(req)
                    self.logger.info("Auto-deactivated expression '%s'.", expression_file)
                except Exception as exc:
                    self.logger.warning("Failed to auto-deactivate '%s': %s", expression_file, exc)
            asyncio.create_task(_auto_deactivate())

        state = "activated" if active else "deactivated"
        return f"Expression '{expression_file}' {state}."

    @tool_function(
        description=(
            "List all Live2D parameters available on the current model, "
            "including their current value, min, max, and default. "
            "Use this to discover what parameters you can control with set_live2d_parameter."
        ),
    )
    async def list_live2d_parameters(self) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(message_type="Live2DParameterListRequest")
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"
        params = response.get("data", {}).get("parameters", [])
        if not params:
            return "No parameters found."
        lines = []
        for p in params:
            name = p.get("name", "?")
            val  = p.get("value", "?")
            mn   = p.get("min", "?")
            mx   = p.get("max", "?")
            df   = p.get("defaultValue", "?")
            lines.append(f"{name}: value={val} (min={mn}, max={mx}, default={df})")
        return "\n".join(lines)

    @tool_function(
        description=(
            "Set one or more Live2D parameter values on the current model. "
            "Use list_live2d_parameters to discover available parameter names. "
            "Values are injected for face_found_seconds seconds before reverting to default. "
            "Pass face_found=False to release control of a parameter."
        ),
        parameter_descriptions={
            "parameter_name": "Name of the Live2D parameter to set (e.g. 'ParamMouthOpenY')",
            "value": "Target value to inject",
            "weight": "Blend weight 0.0–1.0 (1.0 = fully override, default 1.0)",
            "face_found": "True to hold the injected value, False to release (default True)",
        },
    )
    @fire_and_forget
    async def set_live2d_parameter(
        self,
        parameter_name: str,
        value: float,
        weight: float = 1.0,
        face_found: bool = True,
    ) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(
                message_type="InjectParameterDataRequest",
                data={
                    "faceFound": face_found,
                    "mode": "set",
                    "parameterValues": [
                        {
                            "id": parameter_name,
                            "value": value,
                            "weight": weight,
                        }
                    ],
                },
            )
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"
        return f"Parameter '{parameter_name}' set to {value} (weight={weight})."

    @tool_function(
        description=(
            "Enable or disable the avatar's eyes following the mouse cursor. "
            "Call with enabled=True to start tracking, enabled=False to stop."
        ),
        parameter_descriptions={"enabled": "True to follow mouse, False to stop"},
    )
    @fire_and_forget
    async def set_eye_tracking(self, enabled: bool = True) -> str:
        if enabled:
            if self._mouse_task and not self._mouse_task.done():
                self._mouse_tracking_active = True
            else:
                self._mouse_tracking_active = True
                self._mouse_task = asyncio.create_task(self._mouse_tracking_loop())
            return "Eye tracking enabled."
        else:
            self._mouse_tracking_active = False
            return "Eye tracking disabled."

        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        try:
            request = self.create_request(message_type="CurrentModelRequest")
            response = await self._send_recv(request)
        except ConnectionError as e:
            return f"Error: {e}"
        if response.get("messageType") == "APIError":
            return f"Error: {response.get('data', {}).get('message', 'Unknown error')}"
        data = response.get("data", {})
        pos = data.get("modelPosition", {})
        x        = pos.get("positionX", "N/A")
        y        = pos.get("positionY", "N/A")
        rotation = pos.get("rotation",  "N/A")
        size     = pos.get("size",      "N/A")
        name     = data.get("modelName", "unknown")
        return f"Model: {name} | position: ({x}, {y}) | rotation: {rotation}° | size: {size}"




