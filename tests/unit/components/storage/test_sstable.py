"""Unit tests for SSTable."""

import pytest

from happysimulator.components.storage.sstable import SSTable, SSTableStats


class TestSSTableCreation:
    def test_empty_sstable(self):
        sst = SSTable([])
        assert sst.key_count == 0
        assert sst.min_key is None
        assert sst.max_key is None
        assert len(sst) == 0

    def test_single_entry(self):
        sst = SSTable([("a", 1)])
        assert sst.key_count == 1
        assert sst.min_key == "a"
        assert sst.max_key == "a"

    def test_sorts_data_by_key(self):
        sst = SSTable([("c", 3), ("a", 1), ("b", 2)])
        assert sst.min_key == "a"
        assert sst.max_key == "c"

    def test_level_and_sequence(self):
        sst = SSTable([("a", 1)], level=2, sequence=5)
        assert sst.level == 2
        assert sst.sequence == 5

    def test_invalid_index_interval(self):
        with pytest.raises(ValueError):
            SSTable([], index_interval=0)

    def test_invalid_bloom_fp_rate(self):
        with pytest.raises(ValueError):
            SSTable([], bloom_fp_rate=0.0)
        with pytest.raises(ValueError):
            SSTable([], bloom_fp_rate=1.0)


class TestSSTableGet:
    def test_get_existing_key(self):
        sst = SSTable([("a", 1), ("b", 2), ("c", 3)])
        assert sst.get("a") == 1
        assert sst.get("b") == 2
        assert sst.get("c") == 3

    def test_get_missing_key(self):
        sst = SSTable([("a", 1), ("c", 3)])
        assert sst.get("b") is None

    def test_get_from_empty(self):
        sst = SSTable([])
        assert sst.get("a") is None

    def test_get_many_keys(self):
        data = [(f"key_{i:04d}", i) for i in range(200)]
        sst = SSTable(data, index_interval=16)
        for key, value in data:
            assert sst.get(key) == value

    def test_get_uses_bloom_filter(self):
        data = [(f"key_{i:04d}", i) for i in range(100)]
        sst = SSTable(data, bloom_fp_rate=0.001)
        # Keys not in the set should usually be caught by bloom filter
        not_found_count = 0
        for i in range(100, 200):
            if not sst.contains(f"key_{i:04d}"):
                not_found_count += 1
        # With 0.1% FP rate, almost all should be filtered
        assert not_found_count > 90


class TestSSTableContains:
    def test_contains_existing(self):
        sst = SSTable([("a", 1), ("b", 2)])
        assert sst.contains("a")
        assert sst.contains("b")

    def test_contains_missing_usually_false(self):
        data = [(f"k{i}", i) for i in range(100)]
        sst = SSTable(data, bloom_fp_rate=0.01)
        # Bloom filter should reject most missing keys
        false_positives = sum(1 for i in range(100, 200) if sst.contains(f"k{i}"))
        assert false_positives < 10  # Very few FPs expected


class TestSSTableScan:
    def test_full_scan(self):
        data = [("a", 1), ("b", 2), ("c", 3)]
        sst = SSTable(data)
        result = sst.scan()
        assert result == [("a", 1), ("b", 2), ("c", 3)]

    def test_range_scan(self):
        data = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        sst = SSTable(data)
        result = sst.scan("b", "d")
        assert result == [("b", 2), ("c", 3)]

    def test_scan_start_only(self):
        data = [("a", 1), ("b", 2), ("c", 3)]
        sst = SSTable(data)
        result = sst.scan(start_key="b")
        assert result == [("b", 2), ("c", 3)]

    def test_scan_end_only(self):
        data = [("a", 1), ("b", 2), ("c", 3)]
        sst = SSTable(data)
        result = sst.scan(end_key="c")
        assert result == [("a", 1), ("b", 2)]

    def test_scan_empty_range(self):
        data = [("a", 1), ("d", 4)]
        sst = SSTable(data)
        result = sst.scan("b", "c")
        assert result == []

    def test_scan_empty_sstable(self):
        sst = SSTable([])
        assert sst.scan() == []


class TestSSTablePageReads:
    def test_page_reads_for_get_existing(self):
        sst = SSTable([("a", 1), ("b", 2)])
        # Bloom says yes -> 2 page reads (index + data)
        assert sst.page_reads_for_get("a") == 2

    def test_page_reads_for_get_missing_bloom_rejects(self):
        data = [(f"key_{i:04d}", i) for i in range(100)]
        sst = SSTable(data, bloom_fp_rate=0.001)
        # A key not in the set should usually be rejected by bloom
        # With very low FP rate, most should return 0
        zero_count = sum(1 for i in range(100, 200)
                        if sst.page_reads_for_get(f"key_{i:04d}") == 0)
        assert zero_count > 90

    def test_page_reads_for_get_empty(self):
        sst = SSTable([])
        assert sst.page_reads_for_get("a") == 0

    def test_page_reads_for_scan(self):
        data = [(f"key_{i:03d}", i) for i in range(100)]
        sst = SSTable(data, index_interval=16)
        reads = sst.page_reads_for_scan("key_000", "key_100")
        assert reads > 0

    def test_page_reads_for_scan_empty(self):
        sst = SSTable([])
        assert sst.page_reads_for_scan("a", "z") == 0


class TestSSTableOverlaps:
    def test_overlapping(self):
        sst1 = SSTable([("a", 1), ("c", 3)])
        sst2 = SSTable([("b", 2), ("d", 4)])
        assert sst1.overlaps(sst2)
        assert sst2.overlaps(sst1)

    def test_non_overlapping(self):
        sst1 = SSTable([("a", 1), ("b", 2)])
        sst2 = SSTable([("c", 3), ("d", 4)])
        assert not sst1.overlaps(sst2)

    def test_adjacent(self):
        sst1 = SSTable([("a", 1), ("b", 2)])
        sst2 = SSTable([("b", 2), ("c", 3)])
        assert sst1.overlaps(sst2)

    def test_empty_no_overlap(self):
        sst1 = SSTable([])
        sst2 = SSTable([("a", 1)])
        assert not sst1.overlaps(sst2)


class TestSSTableStats:
    def test_stats(self):
        data = [(f"key_{i}", i) for i in range(50)]
        sst = SSTable(data, index_interval=16, bloom_fp_rate=0.01)
        stats = sst.stats
        assert isinstance(stats, SSTableStats)
        assert stats.key_count == 50
        assert stats.size_bytes > 0
        assert stats.index_entries > 0
        assert stats.bloom_filter_size_bits > 0

    def test_repr(self):
        sst = SSTable([("a", 1), ("z", 26)], level=1, sequence=3)
        r = repr(sst)
        assert "level=1" in r
        assert "seq=3" in r
        assert "count=2" in r
