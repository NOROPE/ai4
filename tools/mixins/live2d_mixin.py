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
from tools.base_mixin import ToolMixin, tool_function

class Live2DMixin(ToolMixin):
    """Provides tools for controlling Live2D models in VTube Studio."""

    MIXIN_NAME = "live2d"

    def __init__(self, config=None, **kwargs) -> None: # no vol needed, lipsync will be handled by VTube Studio
        super().__init__(config, **kwargs)
        self.websocket_url = self.config.raw.get("vtube_studio_websockets_url", "ws://localhost:8001")
        self.websocket_connection: ClientConnection | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._ws_lock = asyncio.Lock()
        self._reconnect_enabled = bool(self.config.raw.get("vtube_studio_reconnect", True))
        self._reconnect_interval = float(self.config.raw.get("vtube_studio_reconnect_interval", 1.0))
        self.icon_path = self.config.raw.get("live2d_icon_path", "")

        _token_dir = Path(__file__).parent / "data" / self.config.name
        _token_dir.mkdir(parents=True, exist_ok=True)
        self._token_file = _token_dir / "vtube_token.txt"

        with open(self.icon_path, "rb") as f:
            self.icon = base64.b64encode(f.read()).decode("utf-8")

        


    async def _send_recv(self, request: str) -> dict:
        """Send a request and receive a response, serialized via lock."""
        if self.websocket_connection is None:
            raise ConnectionError("WebSocket connection not established.")
        async with self._ws_lock:
            await self.websocket_connection.send(request)
            return json.loads(await self.websocket_connection.recv())

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
            self.websocket_connection = await websockets.connect(self.websocket_url)
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

    async def _reconnect_loop(self):
        """Polls the API every `_reconnect_interval` seconds and restarts setup() if the connection is lost."""
        while True:
            await asyncio.sleep(self._reconnect_interval)
            if self.websocket_connection is None:
                self.logger.warning("WebSocket connection is None. Attempting to reconnect...")
                asyncio.create_task(self.setup())  # spawn independently to avoid self-cancellation
                return
            try:
                request = self.create_request(message_type="APIStateRequest")
                response = await self._send_recv(request)
                if response.get("messageType") == "APIError":
                    raise ConnectionError(f"Health check returned API error: {response.get('data', {}).get('message', 'Unknown')}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.warning(f"VTube Studio connection lost ({e}). Reconnecting...")
                asyncio.create_task(self.setup())  # spawn independently to avoid self-cancellation
                return



    async def teardown(self):
        """Close WebSocket connection."""
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
        description="Get the current status of the Live2D model, including its position, rotation, and size.",
        parameter_descriptions={},
    )
    async def get_model_status(self) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(message_type="CurrentModelRequest")
        response = await self._send_recv(request)
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
        description="Move the Live2D model to a specific (x, y) position on screen.",
        parameter_descriptions={
            "x": "Horizontal position (-1000 to 1000, screen edges at -1/1, 0 is center)",
            "y": "Vertical position (-1000 to 1000, screen edges at -1/1, 0 is center)",
            "time_in_seconds": "Duration of the movement animation in seconds (0 to 2)",
        },
    )
    async def set_model_position(self, x: float, y: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": False, "positionX": x, "positionY": y}
        )
        await self._send_recv(request)
        return f"Moved model to position ({x}, {y})."

    @tool_function(
        description="Rotate the Live2D model to a specific angle in degrees.",
        parameter_descriptions={
            "rotation": "Rotation angle in degrees (-360 to 360, positive = clockwise)",
            "time_in_seconds": "Duration of the rotation animation in seconds (0 to 2)",
        },
    )
    async def set_model_rotation(self, rotation: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": False, "rotation": rotation}
        )
        await self._send_recv(request)
        return f"Rotated model to {rotation} degrees."

    @tool_function(
        description="Resize the Live2D model.",
        parameter_descriptions={
            "size": "Model size (-100 to 100, -100 is smallest, 100 is biggest)",
            "time_in_seconds": "Duration of the resize animation in seconds (0 to 2)",
        },
    )
    async def set_model_size(self, size: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": False, "size": size}
        )
        await self._send_recv(request)
        return f"Resized model to {size}."

    @tool_function(
        description="Move the Live2D model by a relative offset from its current position.",
        parameter_descriptions={
            "dx": "Horizontal offset to add to current position (screen width is ~2 units)",
            "dy": "Vertical offset to add to current position (screen height is ~2 units)",
            "time_in_seconds": "Duration of the movement animation in seconds (0 to 2)",
        },
    )
    async def move_model_relative(self, dx: float, dy: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": True, "positionX": dx, "positionY": dy, "rotation": 0.0, "size": 0.0}
        )
        await self._send_recv(request)
        return f"Moved model by offset ({dx}, {dy})."

    @tool_function(
        description="Rotate the Live2D model by a relative angle offset from its current rotation.",
        parameter_descriptions={
            "rotation": "Rotation offset in degrees to add to current rotation (positive = clockwise)",
            "time_in_seconds": "Duration of the rotation animation in seconds (0 to 2)",
        },
    )
    async def rotate_model_relative(self, rotation: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": True, "positionX": 0.0, "positionY": 0.0, "rotation": rotation, "size": 0.0}
        )
        await self._send_recv(request)
        return f"Rotated model by {rotation} degrees."

    @tool_function(
        description="Resize the Live2D model by a relative size offset from its current size.",
        parameter_descriptions={
            "size": "Size offset to add to current model size (-100 to 100 scale)",
            "time_in_seconds": "Duration of the resize animation in seconds (0 to 2)",
        },
    )
    async def resize_model_relative(self, size: float, time_in_seconds: float = 0.5) -> str:
        if not self.websocket_connection:
            return "Error: Not connected to VTube Studio."
        request = self.create_request(
            message_type="MoveModelRequest",
            data={"timeInSeconds": time_in_seconds, "valuesAreRelativeToModel": True, "positionX": 0.0, "positionY": 0.0, "rotation": 0.0, "size": size}
        )
        await self._send_recv(request)
        return f"Resized model by offset {size}."
    



