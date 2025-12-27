import threading

HEX = '0123456789ABCDEF'

# Internal counter and lock to produce monotonically increasing IDs
_counter = 0
_counter_lock = threading.Lock()
_ID_LENGTH = 12

def get_id() -> str:
    """Return a monotonically increasing hex ID (uppercase).

    IDs are zero-padded to at least `_ID_LENGTH` hex digits but may grow
    in length as the counter increases.
    """
    global _counter
    with _counter_lock:
        value = _counter
        _counter += 1

    # Format as uppercase hexadecimal, zero-padded to _ID_LENGTH (will expand if needed)
    return format(value, f'0{_ID_LENGTH}X')