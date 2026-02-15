"""In-memory replicated log for Raft and Paxos consensus protocols.

Provides a simple append-only log with term tracking, commit tracking,
and truncation support. Used as a lightweight alternative to a full
write-ahead log (WAL) implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogEntry:
    """A single entry in the replicated log.

    Attributes:
        index: 1-based position in the log.
        term: The leader term when this entry was created.
        command: The command to apply to the state machine.
    """

    index: int
    term: int
    command: object


class Log:
    """In-memory replicated log with term and commit tracking.

    Entries use 1-based indexing. The log supports append, truncation,
    and commit index advancement.

    Attributes:
        commit_index: Index of the highest committed entry (0 = none).
    """

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []
        self.commit_index: int = 0

    def append(self, term: int, command: object) -> LogEntry:
        """Append a new entry to the log.

        Args:
            term: The term number for this entry.
            command: The command payload.

        Returns:
            The newly created LogEntry.
        """
        index = len(self._entries) + 1
        entry = LogEntry(index=index, term=term, command=command)
        self._entries.append(entry)
        return entry

    def append_entry(self, entry: LogEntry) -> None:
        """Append a pre-built entry, overriding its index to maintain sequence."""
        expected_index = len(self._entries) + 1
        # Re-index to maintain consistency
        actual = LogEntry(index=expected_index, term=entry.term, command=entry.command)
        self._entries.append(actual)

    def get(self, index: int) -> LogEntry | None:
        """Get an entry by 1-based index.

        Args:
            index: 1-based log position.

        Returns:
            The LogEntry at that index, or None if out of range.
        """
        if index < 1 or index > len(self._entries):
            return None
        return self._entries[index - 1]

    def truncate_from(self, index: int) -> int:
        """Remove all entries from the given index onward (inclusive).

        Args:
            index: 1-based index to start truncation from.

        Returns:
            Number of entries removed.
        """
        if index < 1 or index > len(self._entries):
            return 0
        removed = len(self._entries) - (index - 1)
        self._entries = self._entries[: index - 1]
        # Adjust commit_index if it was beyond the truncation point
        if self.commit_index >= index:
            self.commit_index = index - 1
        return removed

    def entries_after(self, index: int) -> list[LogEntry]:
        """Return all entries with index > the given index.

        Args:
            index: Return entries after this 1-based index. Use 0 for all.

        Returns:
            List of LogEntry objects.
        """
        if index < 0:
            index = 0
        return list(self._entries[index:])

    def entries_from(self, index: int) -> list[LogEntry]:
        """Return all entries with index >= the given index.

        Args:
            index: Return entries from this 1-based index.

        Returns:
            List of LogEntry objects.
        """
        if index < 1:
            index = 1
        return list(self._entries[index - 1 :])

    @property
    def last_index(self) -> int:
        """Index of the last entry, or 0 if empty."""
        return len(self._entries)

    @property
    def last_term(self) -> int:
        """Term of the last entry, or 0 if empty."""
        if not self._entries:
            return 0
        return self._entries[-1].term

    @property
    def last_entry(self) -> LogEntry | None:
        """The last entry, or None if empty."""
        if not self._entries:
            return None
        return self._entries[-1]

    def committed_entries(self) -> list[LogEntry]:
        """Return all committed entries (index <= commit_index)."""
        return list(self._entries[: self.commit_index])

    def uncommitted_entries(self) -> list[LogEntry]:
        """Return all uncommitted entries (index > commit_index)."""
        return list(self._entries[self.commit_index :])

    def advance_commit(self, new_commit_index: int) -> list[LogEntry]:
        """Advance the commit index and return newly committed entries.

        Args:
            new_commit_index: The new commit index (must be >= current).

        Returns:
            List of entries that became committed.
        """
        if new_commit_index <= self.commit_index:
            return []
        old = self.commit_index
        self.commit_index = min(new_commit_index, len(self._entries))
        return list(self._entries[old : self.commit_index])

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Log(entries={len(self._entries)}, commit_index={self.commit_index})"
