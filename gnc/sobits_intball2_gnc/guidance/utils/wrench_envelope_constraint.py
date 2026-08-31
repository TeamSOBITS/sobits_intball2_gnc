#!/usr/bin/env python3
"""``toppra`` constraint for a constant (path-independent) wrench envelope.

``ToppraTrajectory``'s wrench-envelope constraint (``F_ENV @ w <= g_ENV``,
``w = inv_dyn(q, qd, qdd)``) is constant along the whole path -- the vehicle's
actuation envelope does not depend on where it is on the path. Plain
``toppra.constraint.SecondOrderConstraint`` has no way to express that: its
``constraint_F``/``constraint_g`` callables are called once per gridpoint
(``compute_constraint_params``), so a constant ``F``/``g`` still gets
rebuilt/copied ``len(gridpoints)`` times, dominating construction time
(measured ~0.9s of ~0.95s total for the real fan geometry's 9951-facet
envelope -- see
``docs/archive/achieved/2026-08-30_toppra_replanning_sd_start_speed_investigation.md``
追記1).

``toppra`` already has a fast path for exactly this case -- ``identical=True``
in ``canlinear_colloc_to_interpolate`` builds ``F``/``g`` once instead of once
per gridpoint (the same path ``toppra.constraint.joint_torque
.JointTorqueConstraint`` already uses for its box-shaped torque limits). This
class is the same pattern generalized from a box ``[I; -I]`` polytope to an
arbitrary half-space polytope, so it applies to the wrench envelope's actual
shape (see ``sobits_intball2_gnc.guidance.utils.actuation_envelope
.wrench_envelope_halfspaces``).

Prototyped and speed/behavior-validated in ``test/
experiment_toppra_identical_constraint.py`` (that doc's 追記2〜3): construction
time for the real fan geometry dropped 1.042s -> 0.172s (~6x), with
``trajectory.duration`` matching the plain-``SecondOrderConstraint`` baseline
exactly (diff=0.0) across all tested cases -- a pure speedup, not a behavior
change.
"""
import numpy as np
from toppra.constraint import DiscretizationType
from toppra.constraint.linear_constraint import (
    LinearConstraint,
    canlinear_colloc_to_interpolate,
)


class WrenchEnvelopeConstraint(LinearConstraint):
    """``F_env @ w <= g_env`` polytope constraint on ``w = inv_dyn(q, qd,
    qdd)``, constant along the whole path.

    Args:
        inv_dyn: callable ``(q, qd, qdd) -> w`` (the wrench for a given path
            state/derivatives), same contract ``SecondOrderConstraint``
            expects.
        F_env: ``(k, dof)`` half-space normal matrix.
        g_env: ``(k,)`` half-space offsets, i.e. the constraint is
            ``F_env @ w <= g_env``.
        dof: dimension of ``w`` (path dof).
        discretization_scheme: passed through to
            :meth:`LinearConstraint.set_discretization_type`; only
            ``Interpolation`` (the default, and what ``ToppraTrajectory``
            uses) is implemented here.
    """

    def __init__(self, inv_dyn, F_env, g_env, dof,
                 discretization_scheme=DiscretizationType.Interpolation):
        super().__init__()
        self.inv_dyn = inv_dyn
        self.F_env = np.asarray(F_env, dtype=float)
        self.g_env = np.asarray(g_env, dtype=float)
        self.dof = dof
        self.set_discretization_type(discretization_scheme)
        self.identical = True
        self._format_string = "    Kind: Wrench envelope constraint (identical F/g)\n"

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path "
                "dof ({:d})".format(self.dof, path.dof)
            )
        v_zero = np.zeros(path.dof)
        p_vec = path(gridpoints)
        ps_vec = path(gridpoints, 1)
        pss_vec = path(gridpoints, 2)

        c_vec = np.array([self.inv_dyn(_p, v_zero, v_zero) for _p in p_vec])
        a_vec = np.array(
            [self.inv_dyn(_p, v_zero, _ps) for _p, _ps in zip(p_vec, ps_vec)]
        ) - c_vec
        b_vec = np.array([
            self.inv_dyn(_p, _ps, pss_)
            for _p, _ps, pss_ in zip(p_vec, ps_vec, pss_vec)
        ]) - c_vec

        if self.discretization_type == DiscretizationType.Collocation:
            return a_vec, b_vec, c_vec, self.F_env, self.g_env, None, None
        if self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                a_vec, b_vec, c_vec, self.F_env, self.g_env, None, None,
                gridpoints, identical=True,
            )
        raise NotImplementedError("Other form of discretization not supported!")
