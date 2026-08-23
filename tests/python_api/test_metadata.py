import re
from pathlib import Path

import mlite


def project_metadata():
    content = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    result = {}
    for field in ("name", "version", "description"):
        match = re.search(rf'^\s*{field}\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert match is not None
        result[field] = match.group(1)
    return result


def test_library_metadata_comes_from_pyproject():
    metadata = project_metadata()

    assert mlite.__package_name__ == metadata["name"]
    assert mlite.__version__ == metadata["version"]
    assert mlite.__description__ == metadata["description"]
