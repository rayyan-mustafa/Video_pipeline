"""Phase 5: live mode.

Capture, clause detection, live metrics, and a throttled one-line coach, with a
browser HUD as the view. Audio is captured by THIS process, not the browser -
see the note at the top of `session.py` for why that matters.
"""

from .clause import Boundary, ClauseDetector          # noqa: F401
from .coach import LIVE_SYSTEM_PROMPT, LiveCoach, build_live_payload  # noqa: F401
from .monitor import LiveMetrics, SessionState, measure  # noqa: F401
from .ring import Ring                                # noqa: F401
