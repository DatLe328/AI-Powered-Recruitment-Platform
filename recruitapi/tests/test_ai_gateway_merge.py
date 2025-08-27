from ml.apis import _merge_jd

def test_merge_jd():
    assert _merge_jd("req", "desc") == "req\n\ndesc"
    assert _merge_jd("req", "") == "req"
    assert _merge_jd("", "desc") == "desc"
    assert _merge_jd(None, None) == ""
