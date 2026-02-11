"""Tests for MerkleTree."""

import pytest

from happysimulator.sketching.merkle_tree import KeyRange, MerkleNode, MerkleTree


class TestKeyRange:
    """Tests for KeyRange."""

    def test_contains_single_key(self):
        """Single-key range contains only that key."""
        kr = KeyRange(start="b", end="b")
        assert kr.contains("b")
        assert not kr.contains("a")
        assert not kr.contains("c")

    def test_contains_range(self):
        """Range contains keys within bounds."""
        kr = KeyRange(start="b", end="d")
        assert kr.contains("b")
        assert kr.contains("c")
        assert kr.contains("d")
        assert not kr.contains("a")
        assert not kr.contains("e")


class TestMerkleTreeBuild:
    """Tests for building MerkleTree."""

    def test_empty_tree(self):
        """Empty tree has empty root hash."""
        tree = MerkleTree.build({})
        assert tree.root_hash == ""
        assert tree.size == 0
        assert tree.root is None

    def test_single_key(self):
        """Single-key tree has a leaf root."""
        tree = MerkleTree.build({"a": 1})
        assert tree.size == 1
        assert tree.root is not None
        assert tree.root.is_leaf
        assert tree.root.key_range == KeyRange("a", "a")
        assert tree.root_hash != ""

    def test_two_keys(self):
        """Two-key tree has internal root with two leaf children."""
        tree = MerkleTree.build({"a": 1, "b": 2})
        assert tree.size == 2
        assert tree.root is not None
        assert not tree.root.is_leaf
        assert tree.root.left is not None
        assert tree.root.right is not None
        assert tree.root.left.is_leaf
        assert tree.root.right.is_leaf

    def test_multiple_keys(self):
        """Multiple keys produce a balanced tree."""
        tree = MerkleTree.build({"a": 1, "b": 2, "c": 3, "d": 4})
        assert tree.size == 4
        assert tree.root is not None
        assert tree.root.key_range == KeyRange("a", "d")

    def test_deterministic_build(self):
        """Same data always produces the same root hash."""
        data = {"x": 10, "y": 20, "z": 30}
        tree1 = MerkleTree.build(data)
        tree2 = MerkleTree.build(data)
        assert tree1.root_hash == tree2.root_hash

    def test_different_values_different_hash(self):
        """Different values produce different root hash."""
        tree1 = MerkleTree.build({"a": 1})
        tree2 = MerkleTree.build({"a": 2})
        assert tree1.root_hash != tree2.root_hash

    def test_different_keys_different_hash(self):
        """Different keys produce different root hash."""
        tree1 = MerkleTree.build({"a": 1})
        tree2 = MerkleTree.build({"b": 1})
        assert tree1.root_hash != tree2.root_hash


class TestMerkleTreeUpdate:
    """Tests for MerkleTree update and remove."""

    def test_update_existing_key(self):
        """Updating a key changes the root hash."""
        tree = MerkleTree.build({"a": 1, "b": 2})
        old_hash = tree.root_hash

        tree.update("a", 99)

        assert tree.root_hash != old_hash
        assert tree.get("a") == 99
        assert tree.size == 2

    def test_update_new_key(self):
        """Adding a new key via update grows the tree."""
        tree = MerkleTree.build({"a": 1})

        tree.update("b", 2)

        assert tree.size == 2
        assert tree.get("b") == 2

    def test_remove_key(self):
        """Removing a key shrinks the tree."""
        tree = MerkleTree.build({"a": 1, "b": 2})

        result = tree.remove("a")

        assert result is True
        assert tree.size == 1
        assert tree.get("a") is None

    def test_remove_missing_key(self):
        """Removing a missing key returns False."""
        tree = MerkleTree.build({"a": 1})

        result = tree.remove("missing")

        assert result is False
        assert tree.size == 1

    def test_remove_last_key(self):
        """Removing the last key produces an empty tree."""
        tree = MerkleTree.build({"a": 1})

        tree.remove("a")

        assert tree.size == 0
        assert tree.root_hash == ""

    def test_update_to_empty_then_add(self):
        """Building up from empty via update works."""
        tree = MerkleTree.build({})
        tree.update("x", 42)
        assert tree.size == 1
        assert tree.get("x") == 42
        assert tree.root_hash != ""


class TestMerkleTreeDiff:
    """Tests for MerkleTree.diff()."""

    def test_identical_trees_no_diff(self):
        """Identical trees have no differences."""
        data = {"a": 1, "b": 2, "c": 3}
        tree1 = MerkleTree.build(data)
        tree2 = MerkleTree.build(data)

        diffs = tree1.diff(tree2)

        assert diffs == []

    def test_empty_vs_empty(self):
        """Two empty trees have no differences."""
        tree1 = MerkleTree.build({})
        tree2 = MerkleTree.build({})

        assert tree1.diff(tree2) == []

    def test_empty_vs_nonempty(self):
        """Empty vs non-empty reports the full range."""
        tree1 = MerkleTree.build({})
        tree2 = MerkleTree.build({"a": 1, "b": 2})

        diffs = tree1.diff(tree2)

        assert len(diffs) == 1
        assert diffs[0] == KeyRange("a", "b")

    def test_nonempty_vs_empty(self):
        """Non-empty vs empty reports the full range."""
        tree1 = MerkleTree.build({"a": 1, "b": 2})
        tree2 = MerkleTree.build({})

        diffs = tree1.diff(tree2)

        assert len(diffs) == 1
        assert diffs[0] == KeyRange("a", "b")

    def test_single_key_differs(self):
        """A single differing key produces a targeted diff."""
        tree1 = MerkleTree.build({"a": 1, "b": 2, "c": 3, "d": 4})
        tree2 = MerkleTree.build({"a": 1, "b": 999, "c": 3, "d": 4})

        diffs = tree1.diff(tree2)

        # Should find the range containing "b"
        assert len(diffs) >= 1
        found_b = any(kr.contains("b") for kr in diffs)
        assert found_b

    def test_multiple_keys_differ(self):
        """Multiple differing keys produce multiple diff ranges."""
        tree1 = MerkleTree.build({"a": 1, "b": 2, "c": 3, "d": 4})
        tree2 = MerkleTree.build({"a": 999, "b": 2, "c": 3, "d": 999})

        diffs = tree1.diff(tree2)

        assert len(diffs) >= 1
        found_a = any(kr.contains("a") for kr in diffs)
        found_d = any(kr.contains("d") for kr in diffs)
        assert found_a
        assert found_d

    def test_all_keys_differ(self):
        """All keys differing reports at least one range."""
        tree1 = MerkleTree.build({"a": 1, "b": 2})
        tree2 = MerkleTree.build({"a": 10, "b": 20})

        diffs = tree1.diff(tree2)

        assert len(diffs) >= 1

    def test_diff_is_symmetric(self):
        """diff(a, b) and diff(b, a) detect the same ranges."""
        tree1 = MerkleTree.build({"a": 1, "b": 2, "c": 3})
        tree2 = MerkleTree.build({"a": 1, "b": 999, "c": 3})

        diffs_ab = tree1.diff(tree2)
        diffs_ba = tree2.diff(tree1)

        assert len(diffs_ab) == len(diffs_ba)


class TestMerkleTreeAccessors:
    """Tests for keys(), items(), get()."""

    def test_keys_sorted(self):
        """keys() returns sorted keys."""
        tree = MerkleTree.build({"c": 3, "a": 1, "b": 2})

        assert tree.keys() == ["a", "b", "c"]

    def test_items_sorted(self):
        """items() returns sorted key-value pairs."""
        tree = MerkleTree.build({"c": 3, "a": 1, "b": 2})

        assert tree.items() == [("a", 1), ("b", 2), ("c", 3)]

    def test_get_existing(self):
        """get() returns value for existing key."""
        tree = MerkleTree.build({"a": 42})

        assert tree.get("a") == 42

    def test_get_missing(self):
        """get() returns None for missing key."""
        tree = MerkleTree.build({"a": 42})

        assert tree.get("missing") is None

    def test_repr(self):
        """repr includes size and hash prefix."""
        tree = MerkleTree.build({"a": 1})
        r = repr(tree)
        assert "size=1" in r
        assert "root_hash=" in r
