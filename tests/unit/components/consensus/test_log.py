"""Tests for Log and LogEntry."""

from happysimulator.components.consensus import Log, LogEntry


class TestEmptyLog:
    """Tests for empty log state."""

    def test_empty_log(self):
        """Empty log has last_index=0, last_term=0, commit_index=0."""
        log = Log()

        assert log.last_index == 0
        assert log.last_term == 0
        assert log.commit_index == 0


class TestAppendAndGet:
    """Tests for appending and retrieving entries."""

    def test_append_and_get(self):
        """Appended entries can be retrieved by index."""
        log = Log()

        e1 = log.append(term=1, command="cmd_a")
        e2 = log.append(term=1, command="cmd_b")
        e3 = log.append(term=2, command="cmd_c")

        assert log.get(1) == e1
        assert log.get(2) == e2
        assert log.get(3) == e3
        assert log.get(1).command == "cmd_a"
        assert log.get(2).command == "cmd_b"
        assert log.get(3).term == 2

    def test_1_based_indexing(self):
        """Log entries use 1-based indexing."""
        log = Log()

        entry = log.append(term=1, command="first")

        assert entry.index == 1
        assert log.get(1) is entry
        assert log.get(0) is None

    def test_get_out_of_range(self):
        """get() returns None for invalid indices."""
        log = Log()
        log.append(term=1, command="x")

        assert log.get(0) is None
        assert log.get(-1) is None
        assert log.get(2) is None
        assert log.get(100) is None


class TestTruncate:
    """Tests for log truncation."""

    def test_truncate_from(self):
        """truncate_from removes entries from index onward and returns count."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=2, command="c")
        log.append(term=2, command="d")

        removed = log.truncate_from(3)

        assert removed == 2
        assert len(log) == 2
        assert log.get(1).command == "a"
        assert log.get(2).command == "b"
        assert log.get(3) is None

    def test_truncate_adjusts_commit_index(self):
        """Commit index is adjusted if it falls beyond the truncation point."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=1, command="c")
        log.advance_commit(3)
        assert log.commit_index == 3

        log.truncate_from(2)

        assert log.commit_index == 1
        assert len(log) == 1


class TestEntriesAfterAndFrom:
    """Tests for entries_after and entries_from."""

    def test_entries_after(self):
        """entries_after returns entries with index > given index."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=2, command="c")

        result = log.entries_after(1)

        assert len(result) == 2
        assert result[0].command == "b"
        assert result[1].command == "c"

    def test_entries_after_zero_returns_all(self):
        """entries_after(0) returns all entries."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")

        result = log.entries_after(0)

        assert len(result) == 2

    def test_entries_from(self):
        """entries_from returns entries with index >= given index."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=2, command="c")

        result = log.entries_from(2)

        assert len(result) == 2
        assert result[0].command == "b"
        assert result[1].command == "c"

    def test_entries_from_one_returns_all(self):
        """entries_from(1) returns all entries."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")

        result = log.entries_from(1)

        assert len(result) == 2


class TestCommit:
    """Tests for commit index advancement."""

    def test_advance_commit(self):
        """advance_commit returns newly committed entries."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=2, command="c")

        newly_committed = log.advance_commit(2)

        assert len(newly_committed) == 2
        assert newly_committed[0].command == "a"
        assert newly_committed[1].command == "b"
        assert log.commit_index == 2

    def test_advance_commit_clamped(self):
        """advance_commit is clamped to the log length."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")

        newly_committed = log.advance_commit(100)

        assert log.commit_index == 2
        assert len(newly_committed) == 2

    def test_committed_and_uncommitted_entries(self):
        """committed_entries and uncommitted_entries partition the log."""
        log = Log()
        log.append(term=1, command="a")
        log.append(term=1, command="b")
        log.append(term=2, command="c")
        log.advance_commit(2)

        committed = log.committed_entries()
        uncommitted = log.uncommitted_entries()

        assert len(committed) == 2
        assert committed[0].command == "a"
        assert committed[1].command == "b"
        assert len(uncommitted) == 1
        assert uncommitted[0].command == "c"


class TestAppendEntry:
    """Tests for append_entry with pre-built entries."""

    def test_append_entry(self):
        """Pre-built entry is re-indexed to maintain log sequence."""
        log = Log()
        log.append(term=1, command="a")

        # Create entry with wrong index - should be re-indexed
        entry = LogEntry(index=99, term=2, command="b")
        log.append_entry(entry)

        assert log.get(2).index == 2
        assert log.get(2).term == 2
        assert log.get(2).command == "b"
        assert log.last_index == 2


class TestLastEntryAndLen:
    """Tests for last_entry property and len()."""

    def test_last_entry(self):
        """last_entry returns the last entry in the log."""
        log = Log()

        assert log.last_entry is None

        log.append(term=1, command="a")
        log.append(term=2, command="b")

        assert log.last_entry.command == "b"
        assert log.last_entry.term == 2

    def test_len(self):
        """len() returns the number of entries in the log."""
        log = Log()

        assert len(log) == 0

        log.append(term=1, command="a")
        assert len(log) == 1

        log.append(term=1, command="b")
        log.append(term=2, command="c")
        assert len(log) == 3
