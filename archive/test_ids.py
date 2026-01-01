import re
from happysimulator.utils import ids


def test_get_id_monotonic():
    # Call get_id a few times and ensure numeric values are strictly increasing
    a = ids.get_id()
    b = ids.get_id()
    c = ids.get_id()

    # IDs should be hex strings using uppercase letters
    assert re.fullmatch(r"[0-9A-F]+", a)
    assert re.fullmatch(r"[0-9A-F]+", b)
    assert re.fullmatch(r"[0-9A-F]+", c)

    ia, ib, ic = int(a, 16), int(b, 16), int(c, 16)
    assert ia < ib < ic


def test_get_id_unique_in_threads():
    # Quick concurrency smoke test: spawn multiple threads and ensure uniqueness
    import threading

    results = []
    def worker():
        results.append(ids.get_id())

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    assert len(set(results)) == 100