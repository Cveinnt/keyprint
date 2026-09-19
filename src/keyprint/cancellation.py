"""Cooperative generation cancellation; never interrupts a model kernel."""
from threading import Event


class _CancellationRequested(Exception):
    pass


def check_cancellation(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise _CancellationRequested("Generation cancellation requested")
