"""Tests for the LLM agent episodic memory system."""

from happysimulator.components.llm_agent.backend import MockLLMBackend
from happysimulator.components.llm_agent.memory import EpisodicMemory, MemoryEntry


class TestMemoryEntry:
    def test_frozen_dataclass(self):
        entry = MemoryEntry(time=1.0, summary="Something happened")
        assert entry.time == 1.0
        assert entry.summary == "Something happened"
        assert entry.valence == 0.0
        assert entry.participants == ()
        assert entry.tags == ()

    def test_with_all_fields(self):
        entry = MemoryEntry(
            time=5.0,
            summary="Bob said hello",
            valence=0.3,
            participants=("Bob",),
            tags=("conversation", "greeting"),
        )
        assert entry.participants == ("Bob",)
        assert "greeting" in entry.tags


class TestEpisodicMemory:
    def test_add_and_recent(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Event 1"))
        mem.add(MemoryEntry(time=2.0, summary="Event 2"))
        mem.add(MemoryEntry(time=3.0, summary="Event 3"))

        recent = mem.recent(2)
        assert len(recent) == 2
        assert recent[0].summary == "Event 2"
        assert recent[1].summary == "Event 3"

    def test_recent_fewer_than_n(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Only one"))
        recent = mem.recent(5)
        assert len(recent) == 1

    def test_buffer_max_size_eviction(self):
        mem = EpisodicMemory(max_buffer_size=3)
        for i in range(5):
            mem.add(MemoryEntry(time=float(i), summary=f"Event {i}"))

        assert len(mem.buffer) == 3
        # Oldest should have been evicted
        summaries = [e.summary for e in mem.buffer]
        assert "Event 0" not in summaries
        assert "Event 4" in summaries

    def test_recall_by_summary(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Bob said hello"))
        mem.add(MemoryEntry(time=2.0, summary="Alice left the room"))
        mem.add(MemoryEntry(time=3.0, summary="Bob asked a question"))

        results = mem.recall("bob")
        assert len(results) == 2

    def test_recall_by_tag(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Event 1", tags=("work",)))
        mem.add(MemoryEntry(time=2.0, summary="Event 2", tags=("personal",)))

        results = mem.recall("work")
        assert len(results) == 1
        assert results[0].summary == "Event 1"

    def test_recall_case_insensitive(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Meeting with Bob"))
        results = mem.recall("MEETING")
        assert len(results) == 1

    def test_recall_no_match(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Something happened"))
        results = mem.recall("nonexistent")
        assert len(results) == 0

    def test_compress_with_mock_backend(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Event 1"))
        mem.add(MemoryEntry(time=2.0, summary="Event 2"))

        backend = MockLLMBackend(default_action="A summary of recent events.")
        mem.compress(backend)

        assert mem.short_term_summary == "A summary of recent events."
        assert mem.compression_count == 1

    def test_compress_archives_previous_summary(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Event 1"))

        backend = MockLLMBackend(default_action="First summary.")
        mem.compress(backend)
        assert mem.short_term_summary == "First summary."

        mem.add(MemoryEntry(time=2.0, summary="Event 2"))
        backend._responses = {}
        backend._default_action = "Second summary."
        mem.compress(backend)

        assert mem.short_term_summary == "Second summary."
        assert "First summary." in mem.long_term_summaries

    def test_compress_empty_buffer_noop(self):
        mem = EpisodicMemory()
        backend = MockLLMBackend()
        mem.compress(backend)
        assert mem.short_term_summary == ""
        assert mem.compression_count == 0

    def test_format_for_prompt(self):
        mem = EpisodicMemory()
        mem.add(MemoryEntry(time=1.0, summary="Event 1"))
        mem.add(MemoryEntry(time=2.0, summary="Event 2"))

        text = mem.format_for_prompt()
        assert "Event 1" in text
        assert "Event 2" in text

    def test_format_for_prompt_with_summaries(self):
        mem = EpisodicMemory()
        mem.short_term_summary = "Recent things happened."
        mem.long_term_summaries = ["Old things happened."]
        mem.add(MemoryEntry(time=1.0, summary="Just now"))

        text = mem.format_for_prompt()
        assert "Recent things happened." in text
        assert "Old things happened." in text
        assert "Just now" in text
