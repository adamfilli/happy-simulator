"""Thread-safe unique ID generation.

Provides monotonically increasing hex IDs for event identification and
tracing. IDs are human-readable, sortable, and guaranteed unique within
a process.
"""

import threading


_counter = 0
_counter_lock = threading.Lock()
_ID_LENGTH = 12


def get_id() -> str:
    """Generate a unique, monotonically increasing hex ID.

    Thread-safe. IDs are uppercase hex strings, zero-padded to 12 digits.
    Format: 000000000001, 000000000002, etc.

    Returns:
        Unique hex string ID.
    """
    global _counter
    with _counter_lock:
        value = _counter
        _counter += 1

    return format(value, f'0{_ID_LENGTH}X')