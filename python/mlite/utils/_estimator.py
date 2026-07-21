

def _clone_estimator(estimator):
    if hasattr(estimator, "get_params"):
        return estimator.__class__(**estimator.get_params())

    return estimator.__class__()