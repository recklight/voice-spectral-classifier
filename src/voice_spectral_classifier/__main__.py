"""Run the command-line interface with ``python -m voice_spectral_classifier``.

Equivalent to the ``voice-spectral-classifier`` console script, and usable straight from a
checkout without installing (``PYTHONPATH=src``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    main()
