"""Tests for write policies."""

import pytest

from happysimulator.components.datastore import (
    WriteThrough,
    WriteBack,
    WriteAround,
)


class TestWriteThrough:
    """Tests for WriteThrough policy."""

    def test_always_writes_through(self):
        """WriteThrough always writes to backing store."""
        policy = WriteThrough()

        assert policy.should_write_through() is True

    def test_never_needs_flush(self):
        """WriteThrough never needs flush."""
        policy = WriteThrough()

        policy.on_write("key1", "value1")

        assert policy.should_flush() is False
        assert policy.get_keys_to_flush() == []


class TestWriteBack:
    """Tests for WriteBack policy."""

    def test_doesnt_write_through(self):
        """WriteBack doesn't write immediately to backing."""
        policy = WriteBack(flush_interval=1.0, max_dirty=10)

        assert policy.should_write_through() is False

    def test_tracks_dirty_keys(self):
        """WriteBack tracks dirty keys."""
        policy = WriteBack(flush_interval=1.0, max_dirty=10)

        policy.on_write("key1", "value1")
        policy.on_write("key2", "value2")

        assert policy.dirty_count == 2
        assert set(policy.get_keys_to_flush()) == {"key1", "key2"}

    def test_triggers_flush_at_max_dirty(self):
        """WriteBack triggers flush at max_dirty."""
        policy = WriteBack(flush_interval=1.0, max_dirty=3)

        policy.on_write("key1", "value1")
        policy.on_write("key2", "value2")
        assert policy.should_flush() is False

        policy.on_write("key3", "value3")
        assert policy.should_flush() is True

    def test_on_flush_clears_dirty(self):
        """on_flush removes keys from dirty set."""
        policy = WriteBack(flush_interval=1.0, max_dirty=10)

        policy.on_write("key1", "value1")
        policy.on_write("key2", "value2")

        policy.on_flush(["key1"])

        assert policy.dirty_count == 1
        assert "key1" not in policy.get_keys_to_flush()

    def test_rejects_invalid_params(self):
        """WriteBack rejects invalid parameters."""
        with pytest.raises(ValueError):
            WriteBack(flush_interval=0, max_dirty=10)

        with pytest.raises(ValueError):
            WriteBack(flush_interval=1.0, max_dirty=0)


class TestWriteAround:
    """Tests for WriteAround policy."""

    def test_writes_through(self):
        """WriteAround writes directly to backing store."""
        policy = WriteAround()

        assert policy.should_write_through() is True

    def test_tracks_keys_to_invalidate(self):
        """WriteAround tracks keys to invalidate from cache."""
        policy = WriteAround()

        policy.on_write("key1", "value1")
        policy.on_write("key2", "value2")

        keys = policy.get_keys_to_invalidate()

        assert set(keys) == {"key1", "key2"}

    def test_get_keys_to_invalidate_clears(self):
        """get_keys_to_invalidate clears the list."""
        policy = WriteAround()

        policy.on_write("key1", "value1")
        policy.get_keys_to_invalidate()

        assert policy.get_keys_to_invalidate() == []

    def test_never_needs_flush(self):
        """WriteAround never needs flush."""
        policy = WriteAround()

        policy.on_write("key1", "value1")

        assert policy.should_flush() is False
