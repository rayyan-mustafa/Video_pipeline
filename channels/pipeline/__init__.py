"""
The CPU half of the stickman video pipeline.

Nothing in this package imports torch, diffusers or any GPU dependency. That is a
hard rule, not a coincidence: the moment one module does, the local half stops
being installable on a laptop and the split has been lost. The GPU stages
(FLUX character, frames, thumbnails) stay in the notebook.

Stage boundaries are files in one working folder -- see paths.py for the contract.
"""

__all__ = [
    "brief", "config", "engine", "evidence", "factcheck",
    "lint", "packaging", "paths", "qc", "readsheet", "retention", "script",
]
