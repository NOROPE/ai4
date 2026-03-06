"""
Mixin for Live2D
uses the VTube Studio WebSocket API to control Live2D models in VTube Studio.
Requires VTube Studio v1.8.0+ and the WebSocket API plugin.
"""

from __future__ import annotations
from typing import Any
import websockets
import json
import asyncio
from tools.base_mixin import ToolMixin, tool_function

class Live2DMixin(ToolMixin):
    """Provides tools for controlling Live2D models in VTube Studio."""

    MIXIN_NAME = "live2d"

    def __init__(self, config=None, volume_queue: asyncio.Queue | None = None, **kwargs) -> None: # note: volume_queue is passed in from the mixin kwargs
        super().__init__(config, **kwargs)
        self.websocket_url = self.config.raw.get("vtube_studio_websockets_url", "ws://localhost:8001")
        self.websocket_connection = None
        self.icon = self.config.raw.get("live2d_icon", "")
        self.auth_token = ""
        self._volume_queue = volume_queue
        self._lipsync_task: asyncio.Task | None = None

        
    def create_request(self, message_type: str, description="", **kwargs) -> str:
        """function that makes a request to the VTube Studio WebSocket API with the given description and additional parameters.
            Extra parameters are passed as keyword arguments and included in the request body.
        """
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": description if description else message_type,
            "messageType": message_type
        }
        request.update(kwargs)
        return json.dumps(request)

    async def authenticate(self) -> str:
        """Authenticate with the VTube Studio WebSocket API and store the auth token."""
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"{self.config.name} - AuthRequest",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": f"{self.config.name} Plugin for VTubeStudio",
                "pluginDeveloper": "Winston King",
                "pluginIcon": f"{self.icon}"
                    }
            }
        if self.websocket_connection:
            await self.websocket_connection.send(json.dumps(request))
            response = await self.websocket_connection.recv()
            response_data = json.loads(response)
            return response_data.get("data", {}).get("authenticationToken", "")
        else:
            raise ConnectionError("WebSocket connection not established. Cannot authenticate.")

    

    async def setup(self):
        """Establish WebSocket connection to VTube Studio."""
        try:
            self.websocket_connection = await websockets.connect(self.websocket_url)
            self.logger.info(f"Connected to VTube Studio at {self.websocket_url}")
            await self.websocket_connection.send(
                json.dumps(
                                
                    {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": f"{self.config.name} - StatusRequest",
                        "messageType": "APIStateRequest"
                    }
                )
            )
            status_response = await self.websocket_connection.recv()
            status = json.loads(status_response)
            self.logger.info(f"API status: \n{status}")
            try:
                self.auth_token = await self.authenticate()
                self.logger.info(f"Authenticated with VTube Studio. Auth token: {self.auth_token}")
            except Exception as e:
                self.auth_token = "" # just in case
                self.logger.error(f"Failed to authenticate with VTube Studio: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                await self.setup()
        except Exception as e:
            self.logger.error(f"Failed to connect to VTube Studio: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
            await self.setup()

    async def teardown(self):
        """Close WebSocket connection."""
        if self.websocket_connection:
            await self.websocket_connection.close()
            self.logger.info("WebSocket connection to VTube Studio closed.")