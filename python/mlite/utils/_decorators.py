from functools import wraps


def _ensure_fitted(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return method(self, *args, **kwargs)
    return wrapper