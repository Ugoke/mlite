from importlib.metadata import metadata as _metadata
from importlib.metadata import PackageNotFoundError


try:
    _package = _metadata("mlite-lib")

    __version__ = _package.get("Version")
    __title__ = _package.get("Name")
    __summary__ = _package.get("Summary")
    __author__ = _package.get("Author")
    __license__ = _package.get("License")

    _project_urls = _package.get_all("Project-URL") or []

    __docs__ = next(
        (
            url.split(", ", 1)[1]
            for url in _project_urls
            if url.lower().startswith("documentation, ")
        ),
        None,
    )
except PackageNotFoundError:
    __version__ = "unknown"
    __title__ = "mlite"
    __summary__ = ""
    __author__ = ""
    __license__ = ""
    __docs__ = None


__all__ = [
    "__version__",
    "__title__",
    "__summary__",
    "__author__",
    "__license__",
    "__docs__",
]