"""tools/mixins/memories_mixin.py

Allows the AI to store important memories.
Memories have: ID, Timestamp, Name, Content, Tags.

Stored in tools/mixins/data/{profile_name}_memories.db.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from tools.base_mixin import ToolMixin, fire_and_forget, tool_function


class MemoryToolMixin(ToolMixin):
    """Provides tools for the AI to create and manage long-term memories."""

    MIXIN_NAME = "memories"

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        data_dir = Path(__file__).parent / "data" / (config.name if config else "default")
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(data_dir / "memories.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id        TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    name      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    tags      TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Tool functions
    # ------------------------------------------------------------------

    @tool_function(
        description=(
            "Create a new long-term memory. Returns the memory ID so you can "
            "reference it later with get_memory, update_memory, or delete_memory."
        ),
        parameter_descriptions={
            "name": "Short title for the memory (e.g. 'User birthday')",
            "content": "Full content / detail to remember",
            "tags": "Comma-separated tags for categorisation (e.g. 'personal,dates')",
        },
    )
    async def create_memory(self, name: str, content: str, tags: str = "") -> str:
        memory_id = str(uuid.uuid1()) # not completely random bc i am scared of 1*10^-100 chances and this is guaranteed.
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO memories (id, timestamp, name, content, tags) VALUES (?, ?, ?, ?, ?)",
                (memory_id, time.time(), name, content, tags),
            )
            conn.commit()
        self.logger.info("Memory created: %s (%s)", memory_id, name)
        return memory_id

    @tool_function(
        description="Retrieve a specific memory by its ID.",
        parameter_descriptions={"memory_id": "The ID returned by create_memory"},
    )
    async def get_memory(self, memory_id: str) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, timestamp, name, content, tags FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return f"No memory found with id '{memory_id}'."
        return json.dumps(
            {"id": row[0], "timestamp": row[1], "name": row[2], "content": row[3], "tags": row[4]},
            ensure_ascii=False,
        )

    @tool_function(
        description=(
            "Search memories by keyword. Returns up to top_k results ranked by "
            "how many query words appear in the name, content, or tags."
        ),
        parameter_descriptions={
            "query": "Keywords to search for",
            "top_k": "Maximum number of results to return (default 5)",
        },
    )
    async def search_memories(self, query: str, top_k: int = 5) -> str:
        terms = query.lower().split()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, name, content, tags FROM memories"
            ).fetchall()

        def _score(row: tuple) -> int:
            haystack = f"{row[2]} {row[3]} {row[4]}".lower()
            return sum(t in haystack for t in terms)

        ranked = sorted(rows, key=_score, reverse=True)
        results = [
            {"id": r[0], "timestamp": r[1], "name": r[2], "content": r[3], "tags": r[4]}
            for r in ranked[:top_k]
            if _score(r) > 0
        ]
        if not results:
            return "No memories matched the query."
        return json.dumps(results, ensure_ascii=False)

    @tool_function(
        description="Update an existing memory. Pass only the fields you want to change; omit the rest.",
        parameter_descriptions={
            "memory_id": "ID of the memory to update",
            "name": "New title (leave empty to keep current)",
            "content": "New content (leave empty to keep current)",
            "tags": "New comma-separated tags (leave empty to keep current)",
        },
    )
    @fire_and_forget
    async def update_memory(
        self,
        memory_id: str,
        name: str = "",
        content: str = "",
        tags: str = "",
    ) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT name, content, tags FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return f"No memory found with id '{memory_id}'."
            new_name = name or row[0]
            new_content = content or row[1]
            new_tags = tags or row[2]
            conn.execute(
                "UPDATE memories SET name = ?, content = ?, tags = ? WHERE id = ?",
                (new_name, new_content, new_tags, memory_id),
            )
            conn.commit()
        self.logger.info("Memory updated: %s", memory_id)
        return f"Memory '{memory_id}' updated."

    @tool_function(
        description="Permanently delete a memory by its ID.",
        parameter_descriptions={"memory_id": "ID of the memory to delete"},
    )
    @fire_and_forget
    async def delete_memory(self, memory_id: str) -> str:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        self.logger.info("Memory deleted: %s", memory_id)
        return f"Memory '{memory_id}' deleted."

