import ast
from importlib import resources


STUB_FILES = {
    "__init__.pyi",
    "dtype.pyi",
    "errors.pyi",
    "execution.pyi",
    "metadata.pyi",
    "ops.pyi",
    "tensor.pyi",
}


def test_typing_stubs_are_packaged_by_responsibility():
    package = resources.files("mlite")
    typing_package = package.joinpath("_typing")

    assert package.joinpath("__init__.pyi").is_file()
    assert not package.joinpath("_core.pyi").is_file()
    assert {
        item.name for item in typing_package.iterdir() if item.name.endswith(".pyi")
    } == STUB_FILES


def test_typing_stubs_have_valid_python_syntax():
    typing_package = resources.files("mlite").joinpath("_typing")

    for filename in STUB_FILES:
        stub = typing_package.joinpath(filename)
        ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))


def test_typing_stubs_cover_runtime_protocol_and_executor_configuration():
    package = resources.files("mlite")
    tensor_stub = package.joinpath("_typing", "tensor.pyi").read_text(encoding="utf-8")
    execution_stub = package.joinpath("_typing", "execution.pyi").read_text(
        encoding="utf-8"
    )
    public_stub = package.joinpath("__init__.pyi").read_text(encoding="utf-8")

    assert "def __array__(" in tensor_stub
    assert "def set_num_threads(thread_count: int) -> None" in execution_stub
    assert "set_num_threads as set_num_threads" in public_stub
