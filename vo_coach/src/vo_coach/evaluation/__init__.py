"""Golden-set evaluation: deciding between models on evidence.

`mentor-audit` measures whether the model obeys its rules on whatever takes
happened to be recorded. That is a real signal, but it is not a comparison -
two models never see the same inputs, so a difference in their rates could be a
difference in the takes.

This package fixes the inputs. The same eight cases go to every model, each one
isolating one judgment the coaching could get wrong, and each carrying explicit
expectations about what the reply must and must not do. Running two models over
the same set produces two scores that mean the same thing.

Costs about $0.01 for a full run at the default model, and the mentor cache
makes a repeat run free.
"""

from .cases import CASES, Case, Expect          # noqa: F401
from .runner import CaseResult, EvalRun, render, run_all, run_case  # noqa: F401
