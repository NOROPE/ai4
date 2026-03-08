from openai import AsyncOpenAI
from config_loader import ProfileConfig
import tools
import asyncio
import logging
logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, base_url: str, config: ProfileConfig):
        self.client = AsyncOpenAI(
            base_url=base_url,
            **config.get("client_args", {}),
        )
        self.model = config.get("model", "gpt-4")
        self.history: list[dict] = []  # persistent session history
        self.tools = tools.Tools(config.tool_mixins, config=config)
        asyncio.run(self.tools.setup())

        self.system_instructions = config.system_instruction
        if self.system_instructions:
            self.history.append({"role": "system", "content": self.system_instructions})
        self.prev_chat_memory_path = os.path.join("logs", f"{config.get('name', 'default')}", "transcriptions.json")
        if self.prev_chat_memory_path:
            try:
                with open(self.prev_chat_memory_path, "r") as f:
                    import json
                    prev_history = json.load(f)
                    self.history.extend(prev_history)
            except Exception as e:
                logger.warning(f"Failed to load chat memory from {self.prev_chat_memory_path}: {e}")
        
    def reset(self):
        """Clear the session (like closing and reopening a Live session)."""
        self.history.clear()

    async def send(self, text: str, output_queue: asyncio.Queue):
        """Send a user turn; stream tokens into output_queue.
        
        Puts str chunks into the queue. Puts None when done.
        """
        self.history.append({"role": "user", "content": text})

        kwargs = dict(
            model=self.model,
            messages=self.history,
            stream=True,
        )
        if self.tools:
            kwargs["tools"] = self.tools

        assistant_text = ""
        async with self.client.chat.completions.stream(**kwargs) as stream:
            async for event in stream:
                delta = event.choices[0].delta if event.choices else None
                if delta is None:
                    continue

                # Tool call
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            result = await self.handle_tool_call(
                                tc.function.name, tc.function.arguments
                            )
                            self.history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            })

                # Text token
                if delta.content:
                    assistant_text += delta.content
                    await output_queue.put(delta.content)

        self.history.append({"role": "assistant", "content": assistant_text})
        await output_queue.put(None)  # sentinel: turn complete

    async def handle_system_message(self, text: str):
        """Optionally handle system messages (not currently used, but could be for future extensions like dynamic tool updates or special instructions"""
        self.history.append({"role": "system", "content": text})


    async def handle_tool_call(self, tool_name: str, args: dict) -> str:
        # Override this with actual tool dispatch
        return f"Handled tool call: {tool_name} with args {args}"