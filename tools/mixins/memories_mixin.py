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
    IMPORTANT_MEMORY_LIMIT = 15

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
                    tags      TEXT NOT NULL DEFAULT '',
                    important INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Migrate existing DBs that lack the important column
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            if "important" not in existing_cols:
                conn.execute("ALTER TABLE memories ADD COLUMN important INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    # ------------------------------------------------------------------
    # Tool functions
    # ------------------------------------------------------------------

    @tool_function(
        description=(
            "Create a new long-term memory. Returns the memory ID so you can "
            "reference it later with get_memory, update_memory, or delete_memory. "
            f"Set important=true to pin this memory to the system prompt (max {IMPORTANT_MEMORY_LIMIT}; "
            "oldest important memory is evicted when the limit is reached)."
        ),
        parameter_descriptions={
            "name": "Short title for the memory (e.g. 'User birthday')",
            "content": "Full content / detail to remember",
            "tags": "Comma-separated tags for categorisation (e.g. 'personal,dates')",
            "important": "Pin to system prompt so it is always visible (default false)",
        },
    )
    async def create_memory(self, name: str, content: str, tags: str = "", important: bool = False) -> str:
        memory_id = str(uuid.uuid1()) # not completely random bc i am scared of 1*10^-100 chances and this is guaranteed.
        with sqlite3.connect(self._db_path) as conn:
            if important:
                # Evict oldest important memory if at the limit
                important_ids = conn.execute(
                    "SELECT id FROM memories WHERE important = 1 ORDER BY timestamp ASC"
                ).fetchall()
                if len(important_ids) >= self.IMPORTANT_MEMORY_LIMIT:
                    oldest_id = important_ids[0][0]
                    conn.execute("DELETE FROM memories WHERE id = ?", (oldest_id,))
                    self.logger.info("Evicted oldest important memory: %s", oldest_id)
            conn.execute(
                "INSERT INTO memories (id, timestamp, name, content, tags, important) VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, time.time(), name, content, tags, 1 if important else 0),
            )
            conn.commit()
        self.logger.info("Memory created: %s (%s) important=%s", memory_id, name, important)
        return memory_id

    def get_important_memories_text(self) -> str:
        """Return a formatted string of all important memories for the system prompt."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT name, content, tags FROM memories WHERE important = 1 ORDER BY timestamp ASC"
            ).fetchall()
        if not rows:
            return ""
        lines = ["[Important memories:"]
        for i, (name, content, tags) in enumerate(rows, 1):
            tag_str = f" [{tags}]" if tags else ""
            lines.append(f"  {i}. {name}{tag_str}: {content}")
        lines.append("]")
        return "\n".join(lines)

    @tool_function(
        description="Retrieve a specific memory by its ID.",
        parameter_descriptions={"memory_id": "The ID returned by create_memory"},
    )
    async def get_memory(self, memory_id: str) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, timestamp, name, content, tags, important FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return f"No memory found with id '{memory_id}'."
        return json.dumps(
            {"id": row[0], "timestamp": row[1], "name": row[2], "content": row[3], "tags": row[4], "important": bool(row[5])},
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
                "SELECT id, timestamp, name, content, tags, important FROM memories"
            ).fetchall()

        def _score(row: tuple) -> int:
            haystack = f"{row[2]} {row[3]} {row[4]}".lower()
            return sum(t in haystack for t in terms)

        ranked = sorted(rows, key=_score, reverse=True)
        results = [
            {"id": r[0], "timestamp": r[1], "name": r[2], "content": r[3], "tags": r[4], "important": bool(r[5])}
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
            "important": "Set or clear the important flag (omit to keep current)",
        },
    )
    @fire_and_forget
    async def update_memory(
        self,
        memory_id: str,
        name: str = "",
        content: str = "",
        tags: str = "",
        important: bool | None = None,
    ) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT name, content, tags, important FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return f"No memory found with id '{memory_id}'."
            new_name = name or row[0]
            new_content = content or row[1]
            new_tags = tags or row[2]
            new_important = row[3] if important is None else (1 if important else 0)
            # Enforce cap when marking a new memory as important
            if new_important and not row[3]:
                important_ids = conn.execute(
                    "SELECT id FROM memories WHERE important = 1 ORDER BY timestamp ASC"
                ).fetchall()
                if len(important_ids) >= self.IMPORTANT_MEMORY_LIMIT:
                    oldest_id = important_ids[0][0]
                    conn.execute("DELETE FROM memories WHERE id = ?", (oldest_id,))
                    self.logger.info("Evicted oldest important memory: %s", oldest_id)
            conn.execute(
                "UPDATE memories SET name = ?, content = ?, tags = ?, important = ? WHERE id = ?",
                (new_name, new_content, new_tags, new_important, memory_id),
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

