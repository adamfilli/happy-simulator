"""Episodic memory system for LLM-powered agents.

Two memory mechanisms: summarize-and-compress for explicit memory,
and state-as-implicit-memory (encoded in HumanState fields).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from happysimulator.components.llm_agent.backend import LLMBackend

_DEFAULT_BUFFER_SIZE = 20


@dataclass(frozen=True)
class MemoryEntry:
    """A single episodic memory.

    Attributes:
        time: Simulation time in seconds when this occurred.
        summary: Natural language description of what happened.
        valence: Emotional weight (-1 negative, +1 positive).
        participants: Names of agents involved.
        tags: Keyword tags for recall.
    """

    time: float
    summary: str
    valence: float = 0.0
    participants: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class EpisodicMemory:
    """Sliding-window episodic buffer with summarization.

    Recent events are stored raw in a bounded deque. Periodically,
    the buffer is compressed into narrative summaries using the LLM.

    Args:
        max_buffer_size: Maximum raw entries to keep.
    """

    def __init__(self, max_buffer_size: int = _DEFAULT_BUFFER_SIZE):
        self.buffer: deque[MemoryEntry] = deque(maxlen=max_buffer_size)
        self.short_term_summary: str = ""
        self.long_term_summaries: list[str] = []
        self._compression_count: int = 0

    def add(self, entry: MemoryEntry) -> None:
        """Add a new memory entry to the buffer."""
        self.buffer.append(entry)

    def recent(self, n: int = 5) -> list[MemoryEntry]:
        """Return the n most recent memories."""
        entries = list(self.buffer)
        return entries[-n:]

    def recall(self, topic: str) -> list[MemoryEntry]:
        """Search buffer for entries matching a topic keyword.

        Performs case-insensitive substring match on summary and tags.
        """
        topic_lower = topic.lower()
        results: list[MemoryEntry] = []
        for entry in self.buffer:
            if topic_lower in entry.summary.lower():
                results.append(entry)
                continue
            for tag in entry.tags:
                if topic_lower in tag.lower():
                    results.append(entry)
                    break
        return results

    def compress(self, backend: LLMBackend) -> None:
        """Summarize buffer into short_term_summary using LLM.

        Moves current short_term_summary to long_term_summaries,
        then compresses recent buffer into a new short_term_summary.
        """
        if not self.buffer:
            return

        # Archive existing short-term summary
        if self.short_term_summary:
            self.long_term_summaries.append(self.short_term_summary)

        # Build compression prompt
        entries_text = "\n".join(
            f"- [{e.time:.1f}s] {e.summary}" for e in self.buffer
        )
        prompt = (
            "Summarize these recent events into a brief narrative "
            "(2-3 sentences):\n\n" + entries_text
        )

        self.short_term_summary = backend.complete(
            prompt, temperature=0.3, max_tokens=200
        )
        self._compression_count += 1

    def format_for_prompt(self, max_entries: int = 5) -> str:
        """Format memory for inclusion in a decision prompt."""
        parts: list[str] = []

        if self.long_term_summaries:
            parts.append("Long-term memory: " + " ".join(self.long_term_summaries[-2:]))

        if self.short_term_summary:
            parts.append("Recent memory: " + self.short_term_summary)

        recent = self.recent(max_entries)
        if recent:
            items = "; ".join(e.summary for e in recent)
            parts.append("Just now: " + items)

        return "\n".join(parts)

    @property
    def compression_count(self) -> int:
        """Number of times compress() has been called."""
        return self._compression_count
