"""Smoke test (stub).

Ensures module entrypoint exists.
"""


def test_main_importable():
    import src.main as m

    assert callable(m.main)
