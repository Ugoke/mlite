import mlite


def test_metadata():
    required = {
        "__version__",
        "__title__",
        "__summary__",
        "__author__",
        "__license__",
        "__docs__",
    }

    assert required.issubset(set(mlite.__all__))

    for name in required:
        assert hasattr(mlite, name)


def test_metadata_values():
    assert isinstance(mlite.__title__, str)
    assert mlite.__title__ == "mlite"

    assert isinstance(mlite.__version__, str)
    assert mlite.__version__ != "unknown"

    assert isinstance(mlite.__summary__, str)
    assert len(mlite.__summary__) > 0

    assert isinstance(mlite.__author__, str)
    assert len(mlite.__author__) > 0

    assert isinstance(mlite.__license__, str)
    assert len(mlite.__license__) > 0


def test_metadata_docs():
    assert mlite.__docs__ is None or isinstance(mlite.__docs__, str)

    if mlite.__docs__:
        assert mlite.__docs__.startswith(("http://", "https://"))