from config_loader import ProfileConfig
from human_ai.llm import LLM
import asyncio
import sys
import logging

logger = logging.getLogger(__name__)
# make loggger hide all msgs
logging.basicConfig(level=logging.CRITICAL)

llm = LLM(
    base_url="http://localhost:1234/v1",
    config=ProfileConfig(
        name="TestBot",
        model="gqwen3.5-9b",
        system_instruction="You are a helpful assistant. Don't think before speaking, just answer the question directly.",
        prevmsg_count=10,
    ),
)

# Single stdin reader — avoids two threads competing for stdin
_stdin_queue: asyncio.Queue[str] = asyncio.Queue()

async def _stdin_reader() -> None:
    """Read lines from stdin and push them onto _stdin_queue."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        await _stdin_queue.put(line.rstrip("\n"))


async def main() -> None:
    asyncio.create_task(_stdin_reader(), name="stdin_reader")

    while True:
        print("User: ", end="", flush=True)
        user_input = await _stdin_queue.get()
        if user_input.lower() in {"exit", "quit"}:
            break

        token_queue: asyncio.Queue = asyncio.Queue()
        llm.generate(user_input, token_queue)
        print("AI: (press Enter to interrupt)", end=" ", flush=True)

        while True:
            # Race: next token vs user pressing Enter
            token_task = asyncio.create_task(token_queue.get())
            stdin_task = asyncio.create_task(_stdin_queue.get())
            done, _ = await asyncio.wait(
                {token_task, stdin_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stdin_task in done:
                # User pressed Enter — interrupt
                token_task.cancel()
                llm.interrupt()
                print("\n--- interrupted ---")
                break

            stdin_task.cancel()
            token = token_task.result()
            if token is llm.end_token:
                print("\n--- turn complete ---")
                break
            elif token is None:
                print("\n--- error ---")
                break
            else:
                print(token, end="", flush=True)

asyncio.run(main())
