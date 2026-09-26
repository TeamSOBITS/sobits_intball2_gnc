#!/usr/bin/env python3
"""Guidance-side TF velocity estimator (ROS-agnostic, pure).

Finite-difference + EMA velocity estimate, independent of
``control.utils.trajectory_controller.TrajectoryController``'s own estimator
(same technique, but a separate instance) -- Guidance has no velocity source
of its own (``TfClient.get_pose()`` returns position only) and the Control
estimator can't be reused because the two packages only have a one-way
Guidance -> Control data flow (see
``docs/guidance_realtime_replanning_design.md`` 3-3 節).

Threading contract (``docs/guidance_velocity_estimator_design.md`` 3/5 節):
``update()`` is meant to be called from a single timer thread (``guidance.py``
drives it via a low-rate ``rclpy`` timer, deliberately slower than Control's
50Hz P+D loop -- see that document's 3-2 節 on frequency separation).
``get()`` may be called from a different thread (the action server's
``execute()`` callback). The two never share mutable state directly: each
``update()`` call builds a brand new :class:`VelocityEstimate` and reassigns
it to a single attribute in one step, which is atomic under CPython's GIL --
so ``get()`` never needs a lock or sees a partially-updated estimate.
"""


class VelocityEstimate:
    """Immutable snapshot of the latest velocity estimate.

    Args:
        pos: last position sample ``[x, y, z]``, or ``None`` before the first
            ``update()``.
        vel: EMA-filtered velocity estimate ``[vx, vy, vz]`` (zeros before
            the first usable sample).
        stamp: the TF stamp ``pos`` was measured at, or ``None`` before the
            first ``update()``.
    """

    __slots__ = ("pos", "vel", "stamp")

    def __init__(self, pos, vel, stamp):
        self.pos = pos
        self.vel = vel
        self.stamp = stamp


class VelocityEstimator:
    """Finite-difference + EMA velocity estimate, fed by successive TF poses.

    Args:
        alpha: EMA blend weight (1.0 = no filtering), same convention as
            ``TrajectoryController.vel_filter_alpha``.
        max_dt: if the gap between two successive stamps exceeds this many
            seconds, the sample is treated as a stale-TF gap (mirrors
            ``guidance.tf_staleness_timeout``, see ``docs/
            guidance_velocity_estimator_design.md`` 5 節) rather than a real
            velocity: that ``update()`` call adopts ``pos``/``stamp`` as a
            fresh baseline without touching the filtered velocity, so a
            stall-then-burst TF delivery pattern (``docs/archive/achieved/
            recording_cpu_load_control_degradation.md``) can't produce one
            huge spurious finite-difference spike. A backwards-moving stamp
            (e.g. a simulator restart) is handled the same way.
    """

    def __init__(self, alpha=0.3, max_dt=1.0):
        self.alpha = float(alpha)
        self.max_dt = float(max_dt)
        self._latest = VelocityEstimate(pos=None, vel=[0.0, 0.0, 0.0], stamp=None)

    def update(self, pos, stamp):
        """Feed one new TF sample; recompute and store the filtered estimate.

        Call this from one thread only (see module docstring).
        """
        pos = list(pos)
        stamp = float(stamp)
        prev = self._latest

        if prev.pos is None:
            vel = [0.0, 0.0, 0.0]
        else:
            dt = stamp - prev.stamp
            if 0.0 < dt <= self.max_dt:
                vel = [
                    self.alpha * ((pos[i] - prev.pos[i]) / dt)
                    + (1.0 - self.alpha) * prev.vel[i]
                    for i in range(3)
                ]
            else:
                # dt <= 0 (stamp went backwards, e.g. a sim restart) or
                # dt > max_dt (a stale-TF gap): don't fold a bogus
                # finite-difference spike into the filtered estimate, just
                # re-baseline on this sample and hold the last velocity.
                vel = prev.vel

        self._latest = VelocityEstimate(pos=pos, vel=vel, stamp=stamp)

    def get(self):
        """Return the latest :class:`VelocityEstimate`. Safe from any thread."""
        return self._latest

    def set_gains(self, alpha=None, max_dt=None):
        """Update filter gains in place (Category A, see ``docs/archive/
        achieved/2026-08-21_dynamic_parameter_classification.md``) -- picked
        up by the very next ``update()`` call.
        """
        if alpha is not None:
            self.alpha = float(alpha)
        if max_dt is not None:
            self.max_dt = float(max_dt)
