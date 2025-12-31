from dataclasses import dataclass, field
from typing import Any, Generator

from happysimulator.utils.instant import Instant

@dataclass(order=True)
class ProcessContinuation:
    time: Instant
    # Compare only by time, ignore the generator
    generator: Generator = field(compare=False)
    # Identify the entity for debugging/context
    entity: Any = field(compare=False, default=None)