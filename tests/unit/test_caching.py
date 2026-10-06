from finsight.core.caching import DiskCache


def test_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    key = cache.key_for("model", "prompt", {"t": 0})
    assert cache.get(key) is None
    cache.set(key, {"out": "hi"})
    assert cache.get(key) == {"out": "hi"}


def test_key_is_order_insensitive_for_dicts():
    assert DiskCache.key_for({"a": 1, "b": 2}) == DiskCache.key_for({"b": 2, "a": 1})