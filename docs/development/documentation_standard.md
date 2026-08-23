# Documentation Standard for the Project

For the complete developer workflow, see [Creating New Modules](new_module_development.md).

## General Principle

Each public component has its own dedicated documentation page.

The documentation must explain:

- what the component is;
- how it works;
- how to use it;

without requiring access to the source code.

Documentation is required for public functionality.

## File Mapping Rule

The documentation structure mirrors the public library structure. A Python source file maps
directly to a Markdown page:

```text
python/mlite/linear_models/linear_regression.py
→ docs/lib/linear_models/linear_regression.md
```

```text
python/mlite/linear_models/logistic_regression.py
→ docs/lib/linear_models/logistic_regression.md
```

For C++, a public component may combine its header and implementation on one page:

```text
cpp/tensor/include/mlite/tensor/tensor.hpp
cpp/tensor/src/tensor.cpp
→ docs/lib/tensor/tensor.md
```

Bindings map to the language-specific documentation subtree:

```text
bindings/python/src/numpy.cpp
→ docs/lib/bindings/python/numpy.md
```

Private helpers do not need separate pages. Document them only when their behavior matters to users
or developers extending the public component.

Do not add service metadata to documentation pages. Package metadata belongs only in
`pyproject.toml`.

## Structure of Each Documentation Page

Use the following sections when they add useful information. Omit sections that do not apply.

### 1. Description

A concise explanation of the component, its responsibility, and its architectural layer.

### 2. Mathematics / Logic

The main formula, algorithm, memory model, or underlying principle. Define shape rules, numerical
behavior, complexity, or concurrency semantics where relevant.

### 3. Creation

How to instantiate the object or call the function. Show the public Python API first, followed by
C++ only when it is part of the supported public interface.

### 4. Core Methods

The primary methods and operations, including mutation, ownership, and copying behavior.

### 5. Parameters

Input parameters with accepted types, defaults, validation rules, and units. Tensor strides must
explicitly state whether they are measured in elements or bytes.

### 6. Output

Return type, shape, dtype, device, ownership, layout, and exceptions where applicable.

### 7. Examples

Provide:

1. a minimal executable example;
2. an extended example resembling real usage.

Examples must use only public APIs and must not rely on unspecified behavior.

### 8. Attributes

Public properties and observable state. Do not document private fields as stable API.

### 9. Notes

Limitations, performance characteristics, thread-safety, implementation details, pitfalls, and
links to related components.

## API and Example Rules

- Use the import name `mlite`, not the distribution name `mlite-lib`.
- Keep examples compatible with the minimum supported Python version.
- State whether an operation copies or shares memory.
- State whether an operation supports broadcasting, non-contiguous tensors, and read-only memory.
- State dtype and device restrictions explicitly.
- Document only callable public functionality.
- Use `pytest` for every executable documentation contract.
- Avoid duplicating package name, version, or description. Link to `pyproject.toml` or use runtime
  metadata such as `mlite.__version__`.

## Links and Navigation

Every new page must be linked from [Documentation Home](../../README.md). Use repository-relative
Markdown links and verify that every link resolves.

## Review Checklist

Before merging documentation, verify:

- signatures, defaults, dtype rules, and exceptions match the implementation;
- zero-copy and ownership claims are tested;
- minimal and extended examples are executable;
- the page is linked from an index;
- no package metadata is duplicated outside `pyproject.toml`.
