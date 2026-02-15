"""Batch accumulation and processing.

BatchProcessor collects items until ``batch_size`` is reached or a
``timeout_s`` expires, then processes the entire batch with a single
``process_time`` delay and forwards all items to ``downstream``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

_BATCH_TIMEOUT = "_BatchTimeout"


@dataclass(frozen=True)
class BatchProcessorStats:
    """Snapshot of batch processor statistics."""

    batches_processed: int = 0
    items_processed: int = 0
    timeouts: int = 0


class BatchProcessor(Entity):
    """Entity that accumulates items into batches before processing.

    Items are buffered until ``batch_size`` is reached or ``timeout_s``
    elapses since the first item in the current batch arrived. The batch
    is then processed (yielding ``process_time``) and all items forwarded
    to ``downstream``.

    Args:
        name: Identifier for logging.
        downstream: Entity to receive processed batch items.
        batch_size: Number of items per batch.
        process_time: Seconds to process one complete batch.
        timeout_s: Maximum wait time before flushing a partial batch.
            Use 0 to disable timeout (only flush on full batch).
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        batch_size: int = 10,
        process_time: float = 1.0,
        timeout_s: float = 0.0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if process_time < 0:
            raise ValueError(f"process_time must be >= 0, got {process_time}")
        super().__init__(name)
        self.downstream = downstream
        self.batch_size = batch_size
        self.process_time = process_time
        self.timeout_s = timeout_s

        self._buffer: list[Event] = []
        self._processing = False
        self._timeout_event: Event | None = None

        self._batches_processed = 0
        self._items_processed = 0
        self._timeouts = 0

    @property
    def batches_processed(self) -> int:
        return self._batches_processed

    @property
    def items_processed(self) -> int:
        return self._items_processed

    @property
    def timeouts(self) -> int:
        return self._timeouts

    @property
    def buffer_depth(self) -> int:
        return len(self._buffer)

    @property
    def stats(self) -> BatchProcessorStats:
        return BatchProcessorStats(
            batches_processed=self._batches_processed,
            items_processed=self._items_processed,
            timeouts=self._timeouts,
        )

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event]:
        if event.event_type == _BATCH_TIMEOUT:
            return self._handle_timeout()

        self._buffer.append(event)

        # Schedule timeout when first item enters an empty buffer
        if len(self._buffer) == 1 and self.timeout_s > 0:
            self._timeout_event = Event(
                time=self.now + self.timeout_s,
                event_type=_BATCH_TIMEOUT,
                target=self,
            )
            return [self._timeout_event]

        if len(self._buffer) >= self.batch_size:
            return self._process_batch()

        return []

    def _handle_timeout(self) -> Generator[float, None, list[Event]] | list[Event]:
        self._timeout_event = None
        if not self._buffer:
            return []
        self._timeouts += 1
        return self._process_batch()

    def _process_batch(self) -> Generator[float, None, list[Event]]:
        batch = list(self._buffer)
        self._buffer.clear()

        # Cancel any pending timeout
        if self._timeout_event is not None:
            self._timeout_event.cancel()
            self._timeout_event = None

        self._processing = True
        yield self.process_time
        self._processing = False

        self._batches_processed += 1
        self._items_processed += len(batch)

        # Forward all batch items downstream
        result: list[Event] = []
        for item in batch:
            result.append(
                Event(
                    time=self.now,
                    event_type=item.event_type,
                    target=self.downstream,
                    context=item.context,
                )
            )
        return result
