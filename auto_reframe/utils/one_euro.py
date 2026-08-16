"""
utils/one_euro.py — Phase 4: Adaptive One Euro Filter

A minimal, high-precision, zero-external-dependency implementation of the
One Euro Filter (Casiez, Roussel, Vogel 2012) for video coordinate smoothing.

In accordance with Hackathon Plan Phase 4 Step 8:
- Adaptive cutoff frequency: stiff at low velocity (eliminates jitter),
  responsive at high velocity (eliminates tracking lag).
- Supports per-frame dynamic beta overriding to allow high responsiveness
  during object/pointing transitions and high stability during talking-head speaker blocks.
"""
import math
from typing import Optional


class OneEuroFilter:
    """
    1D One Euro Filter for adaptive low-pass signal smoothing.

    Parameters:
        t0 (float): Initial timestamp.
        x0 (float): Initial signal value.
        min_cutoff (float): Minimum cutoff frequency in Hz (> 0). Controls jitter at low speeds.
        beta (float): Speed coefficient. Controls responsiveness / reduces lag at high speeds.
        d_cutoff (float): Cutoff frequency for derivative estimate in Hz.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def reset(self, t0: float, x0: float) -> None:
        """Resets filter state to initial conditions."""
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    @staticmethod
    def _alpha(cutoff: float, t_e: float) -> float:
        """Computes exponential smoothing factor alpha = 1 / (1 + tau / t_e)."""
        if cutoff <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t: float, x: float, beta: Optional[float] = None) -> float:
        """
        Filters the incoming sample (t, x) and updates internal state.

        Args:
            t: Current sample timestamp in seconds.
            x: Current raw sample value.
            beta: Optional per-frame beta override.
        """
        # Guard against zero or negative delta time
        t_e = max(float(t) - self.t_prev, 1e-6)
        beta_used = self.beta if beta is None else float(beta)

        # Estimate signal derivative and filter it
        a_d = self._alpha(self.d_cutoff, t_e)
        dx = (float(x) - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Dynamically adapt cutoff frequency based on velocity
        cutoff = self.min_cutoff + beta_used * abs(dx_hat)
        a = self._alpha(cutoff, t_e)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat
