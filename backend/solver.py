from __future__ import annotations

import logging
from typing import Tuple, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive damping
# ---------------------------------------------------------------------------

def _choose_damp(Aw: sp.spmatrix) -> float:
    """
    Estimate the condition number of the weighted design matrix and
    return an appropriate ridge-damping value for lsqr.
    """
    try:
        k = min(6, min(Aw.shape) - 1)
        if k < 1:
            return 0.05
        sv = spla.svds(Aw, k=k, return_singular_vectors=False)
        cond = float(sv.max()) / (float(sv.min()) + 1e-12)
        if cond < 1e6:
            return 0.0
        elif cond < 1e8:
            return 0.05
        else:
            return 0.2
    except Exception as exc:
        logger.debug("svds failed (%s); defaulting damp=0.05", exc)
        return 0.05


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_weighted(
    A: sp.spmatrix,
    y: np.ndarray,
    weights: np.ndarray,
    damp: float | None = None,
) -> Tuple[np.ndarray, float, dict]:
    """
    Solve the weighted least-squares problem:
        min  ||W^½ (Ax − y)||²  +  damp² ||x||²

    Parameters
    ----------
    A       : design matrix [n_rows × n_teams]
    y       : target vector [n_rows]
    weights : per-row non-negative weights [n_rows]
    damp    : ridge damping (None → adaptive)

    Returns
    -------
    x       : solution vector [n_teams]
    damp_used : damping value actually applied
    info    : lsqr diagnostic dict
    """
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))

    # Zero-weight rows contribute nothing; sparse diagonal is fine
    W = sp.diags(sqrt_w, format="csr")
    Aw = W @ A
    yw = sqrt_w * y

    if damp is None:
        damp = _choose_damp(Aw)

    result = spla.lsqr(Aw, yw, damp=damp, iter_lim=50_000, atol=1e-10, btol=1e-10)
    x, istop, itn, r1norm = result[0], result[1], result[2], result[3]

    info = {
        "damp": damp,
        "istop": int(istop),
        "iterations": int(itn),
        "r1norm": float(r1norm),
        "n_teams": int(A.shape[1]),
        "n_rows": int(A.shape[0]),
    }
    logger.debug("lsqr solve: damp=%.4f istop=%d itn=%d r1norm=%.3f", damp, istop, itn, r1norm)
    return x, damp, info


# ---------------------------------------------------------------------------
# Convenience: solve OPR + DPR in one call
# ---------------------------------------------------------------------------

def _build_y_dpr(y: np.ndarray, opr: np.ndarray, A: sp.spmatrix) -> np.ndarray:
    """
    Build the DPR target vector.
    For each match pair (rows 2i, 2i+1): target is how many points the
    *opposing* alliance was suppressed below its OPR expectation.
    Pure Points Suppressed — empirically the most resilient FRC defensive metric.
    """
    y_pred = A.dot(opr)
    res = y - y_pred
    y_dpr = np.zeros_like(y)
    for i in range(0, len(y), 2):
        y_dpr[i] = -res[i + 1]
        y_dpr[i + 1] = -res[i]
    return y_dpr


def solve_opr_dpr(
    A: sp.spmatrix,
    A_opp: sp.spmatrix,
    y: np.ndarray,
    y_opp: np.ndarray,
    weights: np.ndarray,
    dpr_weight_power: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (opr_vector, dpr_vector, solver_info).
    Both solves share the same adaptive damp so diagnostics are comparable.
    dpr_weight_power > 1.0 steepens the time-decay curve for the DPR solve,
    giving more recent matches extra influence.
    """
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    W = sp.diags(sqrt_w, format="csr")
    Aw = W @ A
    damp = _choose_damp(Aw)

    opr, _, info_opr = solve_weighted(A, y, weights, damp=damp)

    y_dpr = _build_y_dpr(y, opr, A)
    dpr_weights = weights ** dpr_weight_power if dpr_weight_power != 1.0 else weights
    dpr, _, info_dpr = solve_weighted(A, y_dpr, dpr_weights, damp=damp)

    return opr, dpr, {"opr": info_opr, "dpr": info_dpr}


def solve_wdpr(
    A: sp.spmatrix,
    y: np.ndarray,
    weights: np.ndarray,
    opr: np.ndarray,
    row_opp_oprs: np.ndarray,
    damp: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Weighted DPR: each match row's base weight is multiplied by the raw sum
    of the opposing alliance's OPR values.  Suppressing a 90-OPR alliance
    counts proportionally more than suppressing a 30-OPR alliance.
    OPR must be solved before calling this function.

    Returns (wdpr_vector, solver_info).
    """
    y_dpr = _build_y_dpr(y, opr, A)
    wdpr_weights = weights * row_opp_oprs
    wdpr, _, info = solve_weighted(A, y_dpr, wdpr_weights, damp=damp)
    return wdpr, info


def solve_qdpr(
    A: sp.spmatrix,
    y: np.ndarray,
    weights: np.ndarray,
    opr: np.ndarray,
    row_opp_oprs: np.ndarray,
    avg_opp_opr: float,
    dpr_weight_power: float = 1.5,
    damp: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Quality-adjusted DPR: each match row is weighted by the opponent alliance's
    combined OPR relative to the season average, so suppressing a strong team
    is worth more than suppressing a weak one.  Also applies dpr_weight_power
    for extra recency emphasis.

    Returns (qdpr_vector, solver_info).
    """
    y_dpr = _build_y_dpr(y, opr, A)
    quality_mult = np.clip(row_opp_oprs / max(avg_opp_opr, 1e-6), 0.25, 2.0)
    qdpr_weights = (weights ** dpr_weight_power) * quality_mult
    qdpr, _, info = solve_weighted(A, y_dpr, qdpr_weights, damp=damp)
    return qdpr, info
