# Build, Test, and Release

## Prerequisites

- A C++20 compiler and CMake 3.18 or newer.
- Python 3.9 or newer for the bindings.
- A Python build frontend such as `python -m build` for wheels.

The Python build uses `scikit-build-core` and obtains its native build dependencies from the
isolated environment declared in `pyproject.toml`.

## CMake options

| Option | Default | Purpose |
|---|---:|---|
| `MLITE_BUILD_PYTHON` | `ON` | Build the pybind11 extension. |
| `MLITE_BUILD_TESTS` | `BUILD_TESTING` for top-level builds | Build native C++ tests. |
| `MLITE_ENABLE_SANITIZERS` | `OFF` | Instrument compiled targets and final binaries. |
| `MLITE_INSTALL_CXX` | `ON` | Generate install rules for the C++ SDK. |

When mlite is included as a subdirectory, native tests default to off. Warning and sanitizer flags
are private build settings and do not leak into downstream consumers.

## Native build and test

```console
cmake -S . -B build -DMLITE_BUILD_PYTHON=OFF -DMLITE_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Use `-DCMAKE_BUILD_TYPE=Release` with single-configuration generators. `--config Release` selects
the configuration for Visual Studio, Xcode, and other multi-configuration generators.

## C++ installation

```console
cmake --install build --config Release --prefix /path/to/prefix
```

The `development` install component contains public headers, static libraries, and these CMake
package files:

```text
<prefix>/lib/cmake/mlite/mliteConfig.cmake
<prefix>/lib/cmake/mlite/mliteConfigVersion.cmake
<prefix>/lib/cmake/mlite/mliteTargets.cmake
```

`find_package(mlite CONFIG REQUIRED)` provides `mlite::mlite` plus module-level targets. The
repository's `cmake/package-smoke` project is an out-of-tree consumer used to verify that installed
headers and transitive dependencies are complete.

## Python wheel

```console
python -m pip install --upgrade build
python -m build --wheel
python -m pip install --force-reinstall dist/mlite_lib-*.whl
python -c "import mlite; print(mlite.__version__)"
```

The wheel profile disables native tests and C++ SDK install rules, then installs only the `python`
CMake component: the `_core` extension and the Python package. The C++ SDK remains available from
a normal CMake install without adding headers or static libraries to the wheel.

Run `python -m pytest` after installing the wheel so tests exercise the installed artifact rather
than a build-tree module.

## Sanitizers

```console
cmake -S . -B build-sanitized \
  -DMLITE_BUILD_PYTHON=ON \
  -DMLITE_BUILD_TESTS=ON \
  -DMLITE_ENABLE_SANITIZERS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitized --parallel
ctest --test-dir build-sanitized --output-on-failure
```

GCC and Clang builds use ASan and UBSan with frame pointers. MSVC supports the ASan part only. The
Python module is instrumented explicitly because link flags attached only to static libraries are
insufficient for the final extension.

## Release checklist

1. Update `[project].version` in `pyproject.toml`; CMake reads version, name, and description only
   from that section.
2. Run Python tests against a freshly built and installed wheel.
3. Run native tests, including an ASan/UBSan build on Linux.
4. Install the C++ SDK into an empty prefix and build `cmake/package-smoke` against it.
5. Confirm the wheel contains the Python package and `_core`, but not C++ headers or static
   libraries.
6. Tag only after the operating-system and Python-version CI matrix passes.
