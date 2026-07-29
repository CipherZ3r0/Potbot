"""
Device Resolution — selects the best available compute device for embedding.

Resolves ``"auto"`` to CUDA → MPS → CPU using a lazy torch import so that
loader and chunker modules (which never need torch) remain torch-free.

Usage::

    from ingestion.device import resolve_device

    device = resolve_device("auto")   # e.g. "cuda" on a CUDA machine
    device = resolve_device("cpu")    # always returns "cpu"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Priority order for auto-detection
_AUTO_PRIORITY: tuple[str, ...] = ("cuda", "mps", "cpu")


def resolve_device(preference: str = "auto") -> str:
    """Return the device string to pass to the embedding backend.

    Parameters
    ----------
    preference:
        ``"auto"`` probes CUDA then MPS then falls back to CPU.
        Any other value (``"cuda"``, ``"mps"``, ``"cpu"``) is returned
        verbatim after a basic availability check with a warning if the
        requested device is unavailable.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"``, ``"cpu"``.
    """
    preference = preference.strip().lower()

    if preference == "auto":
        return _auto_detect()

    if preference in ("cuda", "mps"):
        if _is_available(preference):
            logger.info("device.resolve: using requested device=%s", preference)
            return preference
        logger.warning(
            "device.resolve: requested device=%s is not available, falling back to cpu",
            preference,
        )
        return "cpu"

    if preference != "cpu":
        logger.warning(
            "device.resolve: unknown device preference=%r, falling back to cpu",
            preference,
        )

    return "cpu"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _auto_detect() -> str:
    """Probe devices in priority order and return the first available one."""
    for device in _AUTO_PRIORITY:
        if _is_available(device):
            logger.info("device.resolve: auto-selected device=%s", device)
            return device
    # "cpu" is always available, but _is_available("cpu") returns True anyway
    return "cpu"


def _is_available(device: str) -> bool:
    """Return True if *device* is usable on this machine.

    Uses a lazy import of ``torch`` so that this module can be imported
    without torch installed (e.g. in tests that stub the embedder).
    """
    if device == "cpu":
        return True

    try:
        import torch  # type: ignore[import]
    except ImportError:
        logger.debug("device._is_available: torch not installed, cannot check device=%s", device)
        return False

    if device == "cuda":
        return bool(torch.cuda.is_available())

    if device == "mps":
        # torch.backends.mps is only present in PyTorch >= 1.12
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()

    return False
