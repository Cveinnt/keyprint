"""Classify a provider rewrite rejection without importing or loading a model."""


def check_rewrite_rejection(call, unavailable_error):
    try:
        call()
    except unavailable_error as exc:
        return {"status": "expected_rejection", "error_type": type(exc).__name__,
                "source": "constructed provider object; rewriting must remain unavailable"}, False
    except Exception as exc:
        return {"status": "unexpected_error", "error_type": type(exc).__name__}, True
    return {"status": "unexpected_success", "error_type": None,
            "reason": "Provider-object rewriting must reject, not return a rewrite."}, True
