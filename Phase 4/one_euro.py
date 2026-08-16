"""
utils/one_euro.py

A minimal, dependency-free implementation of the One Euro Filter
(Casiez, Roussel, Vogel 2012), per the plan's Phase 4 Step 8: "adaptive
— stiff during slow motion (kills jitter), loose during fast motion
(kills lag)."

Lives in utils/ rather than inlined in smooth_coords.py for the same
reason io_json.py is split out: it's a generic, testable-on-its-own
concern, not pipeline-stage logic.
"""
import math
from typing import Optional


class OneEuroFilter:
    """One-dimensional One Euro Filter.

    beta can be overridden per-call instead of fixed at construction
    time — smooth_coords.py uses this to run a lower beta (more
    smoothing) while inside a 'speaker' block and a higher beta (more
    responsiveness) while inside an 'object' block, frame by frame,
    without needing a separate filter instance per regime.
    """

    def __init__(self, t0: float, x0: float, min_cutoff: float = 1.0,
                 beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    @staticmethod
    def _alpha(cutoff: float, t_e: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t: float, x: float, beta: Optional[float] = None) -> float:
        t_e = max(t - self.t_prev, 1e-6)  # guard div-by-zero on duplicate/out-of-order timestamps
        beta_used = self.beta if beta is None else beta

        # Derivative estimate, itself low-pass filtered at a fixed cutoff.
        a_d = self._alpha(self.d_cutoff, t_e)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # The signal's own cutoff adapts to how fast it's currently moving:
        # near-still -> low cutoff -> heavy smoothing; fast motion -> high
        # cutoff -> filter gets out of the way so it doesn't lag.
        cutoff = self.min_cutoff + beta_used * abs(dx_hat)
        a = self._alpha(cutoff, t_e)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat
