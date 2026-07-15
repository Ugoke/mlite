# Documentation Standard for the Project

For the complete developer workflow for adding a documented module, see [Creating New Modules](new_module_development.md).

## General Principle

Each component of the library has its own dedicated documentation page.

The documentation should explain:

* what the component is
* how it works
* how to use it

without requiring access to the source code.

## File Mapping Rule

Each source code file has a corresponding documentation file:

```
mlite/linear_models/linear_regression.py
→ docs/lib/linear_models/linear_regression.md
```

```
mlite/linear_models/logistic_regression.py
→ docs/lib/linear_models/logistic_regression.md
```

The documentation structure must always mirror the library structure 1:1.

## Structure of Each Documentation Page

### 1. Description

A brief explanation of what the component does.

### 2. Mathematics / Logic

The main formula, algorithm, or underlying principle (if applicable).

### 3. Creation

How to instantiate the object or use the function.

### 4. Core Methods

Main usage methods (e.g., fit, predict, transform, etc.).

### 5. Parameters

Input parameters with their types and descriptions.

### 6. Output

What the model or method returns.

### 7. Examples

A minimal example and an extended example closer to a real-world use case.

### 8. Attributes

Internal or learned parameters of the model.

### 9. Notes

Limitations, implementation details, important considerations, and potential pitfalls.