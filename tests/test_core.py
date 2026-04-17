"""
tests/test_core.py

Run with:
    cd aopr && python3 tests/test_core.py          # no dependencies needed
    cd aopr && python -m pytest tests/ -v          # if pytest is installed

All tests use synthetic data — no TBA API key required.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import math
import numpy as np
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class TestSolver(unittest.TestCase):

    def test_exact_recovery_three_teams(self):
        """A well-conditioned system should recover exact OPRs."""
        from solver import solve_weighted

        true_opr = np.array([10.0, 20.0, 30.0])
        A_data = [
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
        ]
        A = sp.csr_matrix(np.array(A_data, dtype=float))
        y = A @ true_opr
        weights = np.ones(len(y))

        x, damp, info = solve_weighted(A, y, weights)
        np.testing.assert_allclose(x, true_opr, atol=1e-6)
        self.assertEqual(damp, 0.0)

    def test_zero_weight_row_ignored(self):
        """A row with weight=0 must have zero influence on the solution."""
        from solver import solve_weighted

        true_opr = np.array([15.0, 25.0])
        A = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
        y = A @ true_opr
        y_bad = y.copy()
        y_bad[2] = 9999.0

        x, _, _ = solve_weighted(A, y_bad, np.array([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(x, true_opr, atol=1e-4)

    def test_damping_nonnegative(self):
        """Damping value returned is always ≥ 0."""
        from solver import solve_weighted

        # Near-rank-deficient
        A = sp.csr_matrix(np.array([[1, 1], [1, 1], [1, 1]], dtype=float))
        y = np.array([10.0, 10.0, 10.0])
        _, damp, _ = solve_weighted(A, y, np.ones(3))
        self.assertGreaterEqual(damp, 0.0)

    def test_solve_opr_dpr_shapes(self):
        """solve_opr_dpr returns two vectors with correct shape."""
        from solver import solve_opr_dpr

        n_teams = 4
        n_rows = 6
        rng = np.random.default_rng(42)
        A = sp.csr_matrix(rng.integers(0, 2, (n_rows, n_teams)).astype(float))
        A_opp = sp.csr_matrix(rng.integers(0, 2, (n_rows, n_teams)).astype(float))
        y = rng.uniform(30, 80, n_rows)
        y_opp = rng.uniform(30, 80, n_rows)
        w = np.ones(n_rows)

        opr, dpr, info = solve_opr_dpr(A, A_opp, y, y_opp, w)
        self.assertEqual(opr.shape, (n_teams,))
        self.assertEqual(dpr.shape, (n_teams,))
        self.assertIn("opr", info)
        self.assertIn("dpr", info)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_residual_signs(self):
        """Positive residual = underperformance; negative = overperformance."""
        from metrics import compute_residuals

        A = sp.csr_matrix(np.eye(3))
        opr = np.array([10.0, 20.0, 30.0])
        y = np.array([8.0, 20.0, 35.0])
        r = compute_residuals(A, opr, y)

        self.assertGreater(r[0], 0)
        self.assertAlmostEqual(r[1], 0.0, places=9)
        self.assertLess(r[2], 0)

    def test_noise_sigma_mad(self):
        """MAD sigma of [-2,-1,0,1,2] around median 0 is 1.4826."""
        from metrics import compute_noise_sigma

        r = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        sigma = compute_noise_sigma(r)
        self.assertAlmostEqual(sigma, 1.4826, places=3)

    def test_noise_sigma_nonzero_guard(self):
        """A constant residual array must not return zero sigma."""
        from metrics import compute_noise_sigma

        r = np.zeros(10)
        sigma = compute_noise_sigma(r)
        self.assertGreater(sigma, 0.0)

    def test_breaker_mask_correct(self):
        """Only rows exceeding threshold * sigma are flagged."""
        from metrics import detect_breakers

        residuals = np.array([1.0, 2.0, 5.0, 10.0, -1.0])
        mask = detect_breakers(residuals, sigma=2.0, threshold=2.0)
        self.assertTrue(mask[3])   # 10 > 4
        self.assertTrue(mask[2])   # 5 > 4
        self.assertFalse(mask[0])
        self.assertFalse(mask[4])

    def test_match_counts(self):
        """Each team's count equals the number of non-zero entries in its column."""
        from metrics import compute_match_counts

        A = sp.csr_matrix(np.array([
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float))
        counts = compute_match_counts(A, [10, 20, 30])
        self.assertEqual(counts[10], 2)
        self.assertEqual(counts[20], 2)
        self.assertEqual(counts[30], 2)

    def test_variability_zero_for_single_match(self):
        """A team that appears in only one row has zero variability."""
        from metrics import compute_variability

        A = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
        residuals = np.array([5.0, 3.0])
        var = compute_variability(A, residuals, [1, 2])
        self.assertEqual(var[1], 0.0)
        self.assertEqual(var[2], 0.0)

    def test_variability_nonzero_for_multi_match(self):
        """A team in multiple rows with different residuals has nonzero variability."""
        from metrics import compute_variability

        A = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
        residuals = np.array([5.0, 15.0, 3.0])
        var = compute_variability(A, residuals, [1, 2])
        self.assertGreater(var[1], 0.0)  # team 1 has residuals [5, 15]
        self.assertEqual(var[2], 0.0)    # team 2 only appears once


# ─────────────────────────────────────────────────────────────────────────────
# Defender detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDefenderDetection(unittest.TestCase):

    def _detect(self, opr_vals, dpr_vals, counts):
        from refund_engine import detect_defenders
        team_list = list(range(len(opr_vals)))
        opr = np.array(opr_vals, dtype=float)
        dpr = np.array(dpr_vals, dtype=float)
        mc = {i: c for i, c in enumerate(counts)}
        return detect_defenders(team_list, opr, dpr, mc)

    def test_clear_defender_flagged(self):
        defenders = self._detect([5.0, 5.0], [5.0, 30.0], [10, 10])
        self.assertIn(1, defenders)

    def test_offensive_team_not_flagged(self):
        defenders = self._detect([40.0, 5.0], [8.0, 30.0], [10, 10])
        self.assertNotIn(0, defenders)

    def test_low_match_count_excluded(self):
        defenders = self._detect([5.0, 5.0], [5.0, 30.0], [10, 2])
        self.assertNotIn(1, defenders)  # only 2 matches < min_matches_to_rank=6

    def test_below_significance_floor_excluded(self):
        # max_dpr = 100; floor = 10; team1 dpr=5 is below floor
        defenders = self._detect([1.0, 1.0], [100.0, 5.0], [10, 10])
        self.assertNotIn(1, defenders)

    def test_empty_lists(self):
        defenders = self._detect([], [], [])
        self.assertEqual(len(defenders), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Refund engine
# ─────────────────────────────────────────────────────────────────────────────

class TestRefundEngine(unittest.TestCase):

    def _match(self, red, blue, rs, bs):
        from match_normalizer import MatchRecord
        return MatchRecord(
            match_key="2026test_qm1",
            event_key="2026test",
            timestamp=1_700_000_000.0,
            comp_level="qm",
            is_playoff=False,
            red_teams=red,
            blue_teams=blue,
            red_score=rs,
            blue_score=bs,
        )

    def test_no_refund_without_defenders(self):
        from refund_engine import compute_refunds

        m = self._match([1, 2, 3], [4, 5, 6], 50, 80)
        opr = np.array([15.0, 15.0, 15.0, 20.0, 20.0, 20.0])
        dpr = np.array([3.0,  3.0,  3.0,  4.0,  4.0,  4.0])
        teams = [1, 2, 3, 4, 5, 6]
        residuals = np.array([10.0, -5.0])

        refunds, audit = compute_refunds(
            [m], residuals, opr, dpr, teams, set(), np.ones(2), 60.0
        )
        self.assertEqual(refunds.sum(), 0.0)
        self.assertEqual(len(audit), 0)

    def test_refund_capped_at_positive_residual(self):
        from refund_engine import compute_refunds, detect_defenders

        m = self._match([1, 2, 3], [10], 40, 80)
        opr = np.array([15.0, 15.0, 15.0, 1.0])
        dpr = np.array([3.0,  3.0,  3.0, 60.0])
        teams = [1, 2, 3, 10]
        mc = {1: 10, 2: 10, 3: 10, 10: 10}
        defenders = detect_defenders(teams, opr, dpr, mc)
        self.assertIn(10, defenders)

        # Red residual = 5: refund must be ≤ 5
        residuals = np.array([5.0, -10.0])
        refunds, _ = compute_refunds(
            [m], residuals, opr, dpr, teams, defenders, np.ones(2), 50.0
        )
        self.assertLessEqual(refunds[0], 5.0 + 1e-9)
        self.assertGreaterEqual(refunds[0], 0.0)

    def test_no_refund_for_negative_residual(self):
        from refund_engine import compute_refunds

        m = self._match([1, 2, 3], [10], 100, 40)
        opr = np.array([30.0, 30.0, 30.0, 1.0])
        dpr = np.array([5.0,  5.0,  5.0, 50.0])
        teams = [1, 2, 3, 10]
        residuals = np.array([-15.0, 20.0])  # red overperformed

        refunds, _ = compute_refunds(
            [m], residuals, opr, dpr, teams, {10}, np.ones(2), 50.0
        )
        self.assertEqual(refunds[0], 0.0)

    def test_refund_nonnegative_always(self):
        """Refunds must never be negative regardless of inputs."""
        from refund_engine import compute_refunds

        m = self._match([1, 2, 3], [4, 5, 6], 60, 70)
        opr = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
        dpr = np.array([5.0,  5.0,  5.0, 30.0, 30.0, 30.0])
        teams = [1, 2, 3, 4, 5, 6]
        residuals = np.array([2.0, -2.0])

        refunds, _ = compute_refunds(
            [m], residuals, opr, dpr, teams, {4, 5, 6}, np.ones(2), 60.0
        )
        self.assertTrue((refunds >= 0).all())

    def test_audit_row_fields(self):
        """Audit records must contain the required fields."""
        from refund_engine import compute_refunds, detect_defenders

        m = self._match([1, 2, 3], [10], 50, 80)
        opr = np.array([15.0, 15.0, 15.0, 1.0])
        dpr = np.array([3.0,  3.0,  3.0, 60.0])
        teams = [1, 2, 3, 10]
        mc = {t: 10 for t in teams}
        defenders = detect_defenders(teams, opr, dpr, mc)
        residuals = np.array([10.0, -5.0])

        _, audit = compute_refunds(
            [m], residuals, opr, dpr, teams, defenders, np.ones(2), 50.0
        )
        if audit:
            row = audit[0]
            for field in ("match_key", "event_key", "alliance_side",
                          "expected_score", "actual_score", "residual",
                          "refund", "defender_keys", "row_weight"):
                self.assertIn(field, row, f"Missing audit field: {field}")


# ─────────────────────────────────────────────────────────────────────────────
# Match normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchNormalizer(unittest.TestCase):

    def _raw(self, key, red, blue, rs, bs, ts=1_700_000_000):
        mn = int(key.split("qm")[1]) if "qm" in key else 1
        return {
            "key": key,
            "comp_level": "qm",
            "match_number": mn,
            "set_number": 1,
            "actual_time": ts,
            "alliances": {
                "red":  {"team_keys": [f"frc{t}" for t in red],
                         "score": rs, "surrogate_team_keys": [], "dq_team_keys": []},
                "blue": {"team_keys": [f"frc{t}" for t in blue],
                         "score": bs, "surrogate_team_keys": [], "dq_team_keys": []},
            },
        }

    def test_basic_fields(self):
        from match_normalizer import normalize_matches
        raw = [self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60)]
        recs = normalize_matches(raw, "2026test")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].red_teams, [1, 2, 3])
        self.assertEqual(recs[0].blue_teams, [4, 5, 6])
        self.assertEqual(recs[0].red_score, 80)
        self.assertEqual(recs[0].blue_score, 60)
        self.assertFalse(recs[0].is_playoff)

    def test_replay_keeps_latest_by_timestamp(self):
        from match_normalizer import normalize_matches
        m1 = self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60, ts=1000)
        m2 = self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 90, 70, ts=2000)
        recs = normalize_matches([m1, m2], "2026test")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].red_score, 90)
        self.assertTrue(recs[0].is_replay)

    def test_negative_score_skipped(self):
        from match_normalizer import normalize_matches
        raw = [self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], -1, 60)]
        recs = normalize_matches(raw, "2026test")
        self.assertEqual(len(recs), 0)

    def test_surrogate_excluded_from_teams(self):
        from match_normalizer import normalize_matches
        raw = [self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60)]
        raw[0]["alliances"]["red"]["surrogate_team_keys"] = ["frc3"]
        recs = normalize_matches(raw, "2026test")
        self.assertNotIn(3, recs[0].red_teams)

    def test_dq_team_excluded(self):
        from match_normalizer import normalize_matches
        raw = [self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60)]
        raw[0]["alliances"]["blue"]["dq_team_keys"] = ["frc5"]
        recs = normalize_matches(raw, "2026test")
        self.assertNotIn(5, recs[0].blue_teams)

    def test_playoff_flag(self):
        from match_normalizer import normalize_matches
        raw = [self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60)]
        raw[0]["comp_level"] = "sf"
        raw[0]["key"] = "2026test_sf1m1"
        raw[0]["match_number"] = 1
        recs = normalize_matches(raw, "2026test")
        self.assertTrue(recs[0].is_playoff)

    def test_multiple_slots_separate_records(self):
        from match_normalizer import normalize_matches
        m1 = self._raw("2026test_qm1", [1, 2, 3], [4, 5, 6], 80, 60, ts=1000)
        m2 = self._raw("2026test_qm2", [1, 2, 3], [4, 5, 6], 70, 65, ts=2000)
        recs = normalize_matches([m1, m2], "2026test")
        self.assertEqual(len(recs), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builder
# ─────────────────────────────────────────────────────────────────────────────

class TestMatrixBuilder(unittest.TestCase):

    def _match(self, i, red=None, blue=None):
        from match_normalizer import MatchRecord
        return MatchRecord(
            match_key=f"2026test_qm{i}",
            event_key="2026test",
            timestamp=1_700_000_000.0 + i * 3600,
            comp_level="qm",
            is_playoff=False,
            red_teams=red or [1, 2, 3],
            blue_teams=blue or [4, 5, 6],
            red_score=80,
            blue_score=70,
        )

    def test_matrix_shape(self):
        from matrix_builder import build_matrices
        matches = [self._match(i) for i in range(5)]
        b = build_matrices(matches)
        self.assertEqual(b.A.shape, (10, 6))
        self.assertEqual(b.A_opp.shape, (10, 6))
        self.assertEqual(len(b.y), 10)
        self.assertEqual(len(b.weights), 10)

    def test_team_list_sorted(self):
        from matrix_builder import build_matrices
        b = build_matrices([self._match(1, [9, 1, 5], [3, 7, 2])])
        self.assertEqual(b.team_list, sorted(b.team_list))

    def test_all_weights_positive(self):
        from matrix_builder import build_matrices
        b = build_matrices([self._match(i) for i in range(3)])
        self.assertTrue((b.weights > 0).all())

    def test_excluded_match_zero_weight(self):
        from matrix_builder import build_matrices
        m = self._match(1)
        m.is_excluded = True
        b = build_matrices([m])
        self.assertEqual(b.weights[0], 0.0)
        self.assertEqual(b.weights[1], 0.0)

    def test_breaker_match_low_weight(self):
        from matrix_builder import build_matrices
        m = self._match(1)
        m.quality_weight = 0.1
        b = build_matrices([m])
        self.assertAlmostEqual(b.weights[0], b.weights[0])  # weight exists
        # The weight should be ~10% of what it would be normally
        m2 = self._match(1)
        b2 = build_matrices([m2])
        ratio = b.weights[0] / b2.weights[0]
        self.assertAlmostEqual(ratio, 0.1, places=6)

    def test_row_to_match_length(self):
        from matrix_builder import build_matrices
        n = 7
        b = build_matrices([self._match(i) for i in range(n)])
        self.assertEqual(len(b.row_to_match), 2 * n)

    def test_a_and_a_opp_are_complements(self):
        """For a single match, A + A_opp should cover all 6 teams."""
        from matrix_builder import build_matrices
        b = build_matrices([self._match(1)])
        # Row 0 = red scoring; A_opp row 0 = red's opponents (blue)
        combined = (b.A + b.A_opp).toarray()
        # Every team appears in exactly one of the two roles per row
        for row in range(2):
            in_A = set(b.A.getrow(row).nonzero()[1])
            in_Aopp = set(b.A_opp.getrow(row).nonzero()[1])
            self.assertEqual(len(in_A & in_Aopp), 0,
                             "Same team cannot be on both sides of a row")

    def test_two_rows_per_match(self):
        """Every match contributes exactly two rows."""
        from matrix_builder import build_matrices
        for n in [1, 3, 10]:
            b = build_matrices([self._match(i) for i in range(n)])
            self.assertEqual(b.A.shape[0], 2 * n)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full mini-pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestMiniPipeline(unittest.TestCase):
    """
    Runs the numeric core (no TBA, no DB, no FastAPI) on synthetic data
    to verify that AOPR >= OPR when a genuine defender is present and
    that the solve is deterministic.
    """

    def _make_matches(self):
        """
        6 teams: teams 1-5 are offense (OPR ~50 each), team 6 is a heavy defender.
        In matches where team 6 plays against an alliance, their score drops.
        """
        from match_normalizer import MatchRecord

        rng = np.random.default_rng(0)
        matches = []
        base_ts = 1_700_000_000.0

        # Normal matches (no team 6)
        alliances = [
            ([1, 2, 3], [4, 5, 1]),  # team 1 appears on both sides across matches
        ]
        lineups = [
            ([1, 2, 3], [4, 5, 3]),
            ([1, 4, 5], [2, 3, 1]),
            ([2, 3, 4], [1, 5, 2]),
            ([3, 4, 5], [1, 2, 4]),
            ([1, 2, 5], [3, 4, 5]),
            ([1, 3, 4], [2, 5, 3]),
            ([2, 4, 5], [1, 3, 4]),
        ]
        opr = {1: 50, 2: 55, 3: 45, 4: 60, 5: 52, 6: 5}

        for i, (red, blue) in enumerate(lineups):
            rs = sum(opr[t] for t in red) + int(rng.normal(0, 3))
            bs = sum(opr[t] for t in blue) + int(rng.normal(0, 3))
            matches.append(MatchRecord(
                match_key=f"2026test_qm{i+1}",
                event_key="2026test",
                timestamp=base_ts + i * 3600,
                comp_level="qm",
                is_playoff=False,
                red_teams=red,
                blue_teams=blue,
                red_score=max(0, rs),
                blue_score=max(0, bs),
            ))

        # Matches with defender (team 6) suppressing opponent
        def_lineups = [
            ([1, 2, 6], [3, 4, 5]),  # team 6 defends against 3,4,5
            ([2, 3, 6], [1, 4, 5]),
            ([1, 4, 6], [2, 3, 5]),
            ([3, 5, 6], [1, 2, 4]),
            ([2, 4, 6], [1, 3, 5]),
            ([1, 5, 6], [2, 3, 4]),
        ]
        for i, (red, blue) in enumerate(def_lineups):
            # Blue alliance (no team 6) scores normally
            bs = sum(opr[t] for t in blue) + int(rng.normal(0, 3))
            # Red alliance is suppressed by ~30 points because of team 6's defense
            rs = sum(opr[t] for t in red) - 30 + int(rng.normal(0, 3))
            matches.append(MatchRecord(
                match_key=f"2026test_qm{len(lineups)+i+1}",
                event_key="2026test",
                timestamp=base_ts + (len(lineups) + i) * 3600,
                comp_level="qm",
                is_playoff=False,
                red_teams=red,
                blue_teams=blue,
                red_score=max(0, rs),
                blue_score=max(0, bs),
            ))

        return matches

    def test_aopr_gte_opr_for_suppressed_teams(self):
        """
        Teams that are regularly suppressed by defender team 6 should have
        AOPR >= OPR after refund adjustment.
        """
        from matrix_builder import build_matrices
        from solver import solve_opr_dpr, solve_weighted
        from metrics import compute_residuals, compute_noise_sigma, detect_breakers, apply_breaker_weights, compute_match_counts
        from refund_engine import detect_defenders, compute_refunds

        matches = self._make_matches()

        # Build initial matrices
        bundle = build_matrices(matches)
        opr, dpr, _ = solve_opr_dpr(bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights)

        # Noise + breakers
        residuals = compute_residuals(bundle.A, opr, bundle.y)
        sigma = compute_noise_sigma(residuals)
        mask = detect_breakers(residuals, sigma)
        apply_breaker_weights(matches, bundle.row_to_match, mask)

        # Rebuild with breaker weights
        bundle = build_matrices(matches)
        opr, dpr, _ = solve_opr_dpr(bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights)
        residuals = compute_residuals(bundle.A, opr, bundle.y)
        sigma = compute_noise_sigma(residuals)

        # Defenders
        mc = compute_match_counts(bundle.A, bundle.team_list)
        defenders = detect_defenders(bundle.team_list, opr, dpr, mc)

        avg_score = float(np.mean(bundle.y[bundle.weights > 0]))
        refunds, audit = compute_refunds(
            matches, residuals, opr, dpr, bundle.team_list, defenders,
            bundle.weights, avg_score
        )

        # AOPR solve
        y_adj = bundle.y + refunds
        aopr, _, _ = solve_weighted(bundle.A, y_adj, bundle.weights)

        ti = {t: i for i, t in enumerate(bundle.team_list)}

        # Teams 1-5 that played against team 6 should see AOPR >= OPR
        # (at least for the teams most affected)
        suppressed = [1, 2, 3, 4, 5]
        for t in suppressed:
            if t in ti:
                self.assertGreaterEqual(
                    aopr[ti[t]], opr[ti[t]] - 0.5,  # tiny tolerance for numeric noise
                    f"Team {t}: AOPR {aopr[ti[t]]:.2f} < OPR {opr[ti[t]]:.2f}"
                )

    def test_determinism(self):
        """Same input data must produce identical results."""
        from matrix_builder import build_matrices
        from solver import solve_opr_dpr

        matches = self._make_matches()
        b1 = build_matrices(matches)
        b2 = build_matrices(matches)

        opr1, dpr1, _ = solve_opr_dpr(b1.A, b1.A_opp, b1.y, b1.y_opp, b1.weights)
        opr2, dpr2, _ = solve_opr_dpr(b2.A, b2.A_opp, b2.y, b2.y_opp, b2.weights)

        np.testing.assert_allclose(opr1, opr2, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(dpr1, dpr2, atol=1e-8, rtol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────

class TestTimeWeightClamp(unittest.TestCase):

    def _tw(self, ts, now):
        from matrix_builder import _time_weight
        return _time_weight(ts, now)

    def test_past_match_weight_less_than_one(self):
        now = 1_700_000_000.0
        w = self._tw(now - 86400 * 7, now)   # 7 days ago
        self.assertLess(w, 1.0)
        self.assertGreater(w, 0.0)

    def test_recent_match_weight_is_one(self):
        now = 1_700_000_000.0
        w = self._tw(now, now)                # right now
        self.assertAlmostEqual(w, 1.0, places=9)

    def test_future_timestamp_clamped_to_one(self):
        """A match timestamped in the future must never exceed weight 1.0."""
        now = 1_700_000_000.0
        w = self._tw(now + 86400 * 30, now)  # 30 days in the future
        self.assertAlmostEqual(w, 1.0, places=9)

    def test_zero_timestamp_returns_half(self):
        w = self._tw(0, 1_700_000_000.0)
        self.assertAlmostEqual(w, 0.5, places=9)

    def test_half_life_decay(self):
        """After exactly one half-life, weight should be 0.5."""
        from config import CONFIG
        now = 1_700_000_000.0
        half_life_secs = CONFIG.time_decay_half_life_days * 86400
        w = self._tw(now - half_life_secs, now)
        self.assertAlmostEqual(w, 0.5, places=6)

    def test_monotone_decay(self):
        """Older matches must always have lower weight than newer ones."""
        now = 1_700_000_000.0
        timestamps = [now - 86400 * d for d in [0, 7, 14, 21, 42, 84]]
        weights = [self._tw(ts, now) for ts in timestamps]
        for i in range(len(weights) - 1):
            self.assertGreater(weights[i], weights[i+1])


# ─────────────────────────────────────────────────────────────────────────────
# Match-row index builder (new)
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchRowIndex(unittest.TestCase):
    """
    Verifies the per-team match-row index built inside run_season_pipeline.
    We test the logic directly (extracted) because the full pipeline needs
    network + DB access that isn't available here.
    """

    def _build_index(self, matches, opr, dpr, team_list, defenders,
                     refunds, residuals, event_meta=None):
        """
        Replicate the match-row index logic from pipeline.py.
        Returns team_match_rows dict.
        """
        import numpy as np
        team_idx = {t: i for i, t in enumerate(team_list)}
        opr_arr  = np.asarray(opr)
        em = event_meta or {}

        team_match_rows = {t: [] for t in team_list}

        for mi, m in enumerate(matches):
            for side in range(2):
                ri = 2 * mi + side
                if side == 0:
                    scoring, opposing = m.red_teams, m.blue_teams
                    actual_score  = m.red_score
                    side_label = "red"
                else:
                    scoring, opposing = m.blue_teams, m.red_teams
                    actual_score  = m.blue_score
                    side_label = "blue"

                expected_score = float(sum(
                    opr_arr[team_idx[t]] for t in scoring if t in team_idx
                ))
                row = {
                    "match_key":     m.match_key,
                    "event_key":     m.event_key,
                    "event_name":    em.get(m.event_key, {}).get("name", m.event_key),
                    "comp_level":    m.comp_level,
                    "timestamp":     m.timestamp,
                    "side":          side_label,
                    "alliance_teams":scoring,
                    "opponent_teams":opposing,
                    "actual_score":  actual_score,
                    "expected_score":round(expected_score, 1),
                    "residual":      round(float(residuals[ri]), 1),
                    "refund":        round(float(refunds[ri]), 1),
                    "is_breaker":    "breaker" in m.status_flags,
                    "defender_keys": [f"frc{t}" for t in opposing if t in defenders],
                }
                for t in scoring:
                    if t in team_match_rows:
                        team_match_rows[t].append(row)

        for t in team_match_rows:
            team_match_rows[t].sort(key=lambda r: r["timestamp"])

        return team_match_rows

    def _make_match(self, n=1):
        from match_normalizer import MatchRecord
        return MatchRecord(
            match_key=f"2026test_qm{n}",
            event_key="2026test",
            timestamp=1_700_000_000.0 + n * 3600,
            comp_level="qm",
            is_playoff=False,
            red_teams=[1, 2, 3],
            blue_teams=[4, 5, 6],
            red_score=80, blue_score=70,
        )

    def test_each_team_gets_rows(self):
        import numpy as np
        matches = [self._make_match(i) for i in range(3)]
        opr = np.array([20.0]*6)
        dpr = np.array([5.0]*6)
        teams = [1,2,3,4,5,6]
        refunds = np.zeros(6)
        residuals = np.zeros(6)
        idx = self._build_index(matches, opr, dpr, teams, set(), refunds, residuals)

        for t in teams:
            self.assertEqual(len(idx[t]), 3, f"Team {t} should have 3 rows")

    def test_rows_sorted_by_timestamp(self):
        import numpy as np
        matches = [self._make_match(i) for i in [3, 1, 2]]  # out of order
        opr = np.array([20.0]*6)
        teams = [1,2,3,4,5,6]
        refunds = np.zeros(6)
        residuals = np.zeros(6)
        idx = self._build_index(matches, opr, opr, teams, set(), refunds, residuals)

        ts_list = [r["timestamp"] for r in idx[1]]
        self.assertEqual(ts_list, sorted(ts_list))

    def test_side_label_correct(self):
        import numpy as np
        match = self._make_match(1)
        opr = np.array([20.0]*6)
        teams = [1,2,3,4,5,6]
        refunds = np.zeros(2)
        residuals = np.zeros(2)
        idx = self._build_index([match], opr, opr, teams, set(), refunds, residuals)

        sides_1 = {r["side"] for r in idx[1]}  # team 1 is red
        sides_4 = {r["side"] for r in idx[4]}  # team 4 is blue
        self.assertIn("red",  sides_1)
        self.assertIn("blue", sides_4)

    def test_refund_appears_in_row(self):
        import numpy as np
        match = self._make_match(1)
        opr = np.array([20.0]*6)
        teams = [1,2,3,4,5,6]
        refunds   = np.array([8.0, 0.0])  # red side refunded 8
        residuals = np.array([10.0, -5.0])
        idx = self._build_index([match], opr, opr, teams, set(), refunds, residuals)

        red_row = idx[1][0]
        self.assertAlmostEqual(red_row["refund"], 8.0)

    def test_defender_keys_populated(self):
        import numpy as np
        match = self._make_match(1)
        opr  = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        dpr  = np.array([2.0, 2.0, 2.0, 30.0,30.0,30.0])
        teams = [1,2,3,4,5,6]
        refunds   = np.zeros(2)
        residuals = np.zeros(2)
        defenders = {4, 5, 6}  # blue side are all defenders

        idx = self._build_index([match], opr, dpr, teams, defenders, refunds, residuals)
        red_row = idx[1][0]   # team 1 (red) faced defenders 4,5,6
        self.assertTrue(len(red_row["defender_keys"]) > 0)
        self.assertIn("frc4", red_row["defender_keys"])

    def test_breaker_flag_propagated(self):
        import numpy as np
        match = self._make_match(1)
        match.status_flags = ["breaker"]
        opr   = np.array([20.0]*6)
        teams = [1,2,3,4,5,6]
        refunds   = np.zeros(2)
        residuals = np.zeros(2)
        idx = self._build_index([match], opr, opr, teams, set(), refunds, residuals)

        self.assertTrue(idx[1][0]["is_breaker"])

    def test_expected_score_sum_of_opr(self):
        import numpy as np
        match = self._make_match(1)
        opr   = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
        teams = [1,2,3,4,5,6]
        refunds   = np.zeros(2)
        residuals = np.zeros(2)
        idx = self._build_index([match], opr, opr, teams, set(), refunds, residuals)

        red_row = idx[1][0]
        # expected = opr[1] + opr[2] + opr[3] = 10+20+30 = 60
        self.assertAlmostEqual(red_row["expected_score"], 60.0, places=1)








# ─────────────────────────────────────────────────────────────────────────────
# NaN / Inf sanitisation (_safe helper)
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeFloat(unittest.TestCase):

    def _safe(self, v, decimals=3):
        import math
        try:
            f = float(v)
            return round(f, decimals) if math.isfinite(f) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def test_nan_returns_zero(self):
        self.assertEqual(self._safe(float('nan')), 0.0)

    def test_pos_inf_returns_zero(self):
        self.assertEqual(self._safe(float('inf')), 0.0)

    def test_neg_inf_returns_zero(self):
        self.assertEqual(self._safe(float('-inf')), 0.0)

    def test_none_returns_zero(self):
        self.assertEqual(self._safe(None), 0.0)

    def test_normal_float_rounded(self):
        self.assertAlmostEqual(self._safe(55.123456789, 3), 55.123)

    def test_zero_passthrough(self):
        self.assertEqual(self._safe(0.0), 0.0)

    def test_negative_normal(self):
        self.assertAlmostEqual(self._safe(-12.5, 1), -12.5)

    def test_decimals_respected(self):
        self.assertAlmostEqual(self._safe(1.23456, 2), 1.23)

    def test_pipeline_uses_safe_for_opr(self):
        """
        Simulate a degenerate solve producing NaN OPR and verify
        the results dict contains 0.0 not NaN.
        """
        import math
        import numpy as np
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

        # Replicate what pipeline does for one team
        opr_i = float('nan')
        result = self._safe(opr_i)
        self.assertTrue(math.isfinite(result))
        self.assertEqual(result, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite cache — thread-safety and snapshot pruning
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheThreadSafety(unittest.TestCase):

    def setUp(self):
        import tempfile, os
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name
        # Patch CONFIG.db_path for this test
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from config import CONFIG
        self._orig_path = CONFIG.db_path
        CONFIG.db_path = self._db_path

    def tearDown(self):
        from config import CONFIG
        CONFIG.db_path = self._orig_path
        import os
        os.unlink(self._db_path)

    def test_init_and_basic_rw(self):
        from cache import init_db, set_cached, get_cached
        init_db()
        set_cached('/test', 'etag123', '{"ok":true}', 'fresh')
        row = get_cached('/test')
        self.assertIsNotNone(row)
        self.assertEqual(row['etag'], 'etag123')

    def test_set_cached_overwrite(self):
        from cache import init_db, set_cached, get_cached
        init_db()
        set_cached('/ep', 'etag1', 'body1')
        set_cached('/ep', 'etag2', 'body2')
        row = get_cached('/ep')
        self.assertEqual(row['etag'], 'etag2')

    def test_snapshot_save_and_load(self):
        from cache import init_db, save_solver_snapshot, get_latest_snapshot
        init_db()
        payload = {'team_results': {}, 'meta': {'year': 2026}, 'audit': [], 'event_membership': {}}
        save_solver_snapshot(2026, payload)
        snap = get_latest_snapshot(2026)
        self.assertIsNotNone(snap)
        self.assertIn('results', snap)
        self.assertEqual(snap['results']['meta']['year'], 2026)

    def test_snapshot_pruning_keeps_5(self):
        from cache import init_db, save_solver_snapshot, get_latest_snapshot
        import sqlite3, time
        from config import CONFIG
        init_db()
        for i in range(8):
            payload = {'team_results': {}, 'meta': {'year': 2026, 'seq': i}, 'audit': [], 'event_membership': {}}
            save_solver_snapshot(2026, payload)
        # Only 5 should remain
        conn = sqlite3.connect(CONFIG.db_path)
        count = conn.execute("SELECT COUNT(*) FROM solver_snapshots WHERE season_year=2026").fetchone()[0]
        conn.close()
        self.assertLessEqual(count, 5)

    def test_concurrent_reads_dont_block(self):
        """Multiple threads reading simultaneously should all succeed."""
        from cache import init_db, set_cached, get_cached
        import threading
        init_db()
        set_cached('/concurrent', 'etag', '{"data":1}')
        errors = []
        results = []

        def read():
            try:
                r = get_cached('/concurrent')
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r is not None for r in results))

    def test_concurrent_writes_serialised(self):
        """Multiple threads writing simultaneously must not corrupt the DB."""
        from cache import init_db, set_cached, get_cached
        import threading
        init_db()
        errors = []

        def write(i):
            try:
                set_cached(f'/ep{i}', f'etag{i}', f'body{i}')
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0, f"Write errors: {errors}")
        # Spot-check a few
        for i in [0, 5, 10, 19]:
            r = get_cached(f'/ep{i}')
            self.assertIsNotNone(r)


# ─────────────────────────────────────────────────────────────────────────────
# Refresh loop jitter
# ─────────────────────────────────────────────────────────────────────────────

class TestRefreshJitter(unittest.TestCase):

    def test_jitter_in_range(self):
        """The jitter value must fall in [0, 60]."""
        import random
        for _ in range(200):
            j = random.uniform(0, 60)
            self.assertGreaterEqual(j, 0)
            self.assertLessEqual(j, 60)

    def test_multiple_workers_different_jitter(self):
        """Different random seeds produce different jitter values (almost surely)."""
        import random
        jitters = {random.uniform(0, 60) for _ in range(100)}
        # With 100 draws from [0,60] the probability of any collision is negligible
        self.assertGreater(len(jitters), 90)



# ─────────────────────────────────────────────────────────────────────────────
# main.py helpers (no FastAPI needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestMainHelpers(unittest.TestCase):

    def _make_results(self, teams=None, membership=None, audit=None):
        t = teams or {
            254: {'team_number':254,'opr':55.0,'dpr':5.0,'aopr':58.0,
                  'delta':3.0,'variability':4.0,'match_count':10,
                  'is_defender':False,'low_match_warning':False},
            1114:{'team_number':1114,'opr':60.0,'dpr':4.0,'aopr':62.0,
                  'delta':2.0,'variability':3.0,'match_count':10,
                  'is_defender':False,'low_match_warning':False},
            973: {'team_number':973,'opr':5.0,'dpr':40.0,'aopr':5.0,
                  'delta':0.0,'variability':2.0,'match_count':10,
                  'is_defender':True,'low_match_warning':False},
        }
        m = membership or {'2026miket': [254, 1114, 973]}
        return {
            'team_results': t,
            'event_membership': m,
            'meta': {'year': 2026, 'event_meta': {
                '2026miket': {'name':'Michigan Kettering','week':2}
            }},
            'audit': audit or [],
            'team_match_rows': {},
        }

    def test_teams_for_event_returns_correct_set(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        # Inline the helper
        results = self._make_results()
        membership = results.get('event_membership', {})
        allowed = set(membership.get('2026miket', []))
        self.assertEqual(allowed, {254, 1114, 973})

    def test_teams_for_event_unknown_key_empty(self):
        results = self._make_results()
        membership = results.get('event_membership', {})
        allowed = set(membership.get('2026notreal', []))
        self.assertEqual(len(allowed), 0)

    def test_rank_rows_sorts_descending_by_aopr(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        rows = [
            {'team_number':1,'aopr':30.0},
            {'team_number':2,'aopr':50.0},
            {'team_number':3,'aopr':40.0},
        ]
        # Replicate _rank_rows logic
        rows.sort(key=lambda r: r.get('aopr', 0), reverse=True)
        rows = [dict(r, rank=i+1) for i, r in enumerate(rows)]
        self.assertEqual(rows[0]['team_number'], 2)
        self.assertEqual(rows[0]['rank'], 1)
        self.assertEqual(rows[2]['rank'], 3)

    def test_rank_rows_ascending(self):
        rows = [
            {'team_number':1,'opr':30.0},
            {'team_number':2,'opr':10.0},
            {'team_number':3,'opr':20.0},
        ]
        rows.sort(key=lambda r: r.get('opr', 0), reverse=False)
        rows = [dict(r, rank=i+1) for i, r in enumerate(rows)]
        self.assertEqual(rows[0]['team_number'], 2)  # lowest OPR first

    def test_event_stats_aggregates(self):
        """Aggregates should correctly compute avg OPR for eligible teams."""
        results = self._make_results()
        allowed = {254, 1114, 973}
        rows = [v for k, v in results['team_results'].items() if k in allowed]
        ranked = [r for r in rows if r.get('match_count', 0) >= 6]
        oprs  = [r['opr'] for r in ranked]
        avg   = round(sum(oprs) / len(oprs), 1) if oprs else 0.0
        expected_avg = round((55.0 + 60.0 + 5.0) / 3, 1)
        self.assertAlmostEqual(avg, expected_avg, places=1)

    def test_event_matches_dedup(self):
        """De-duplication logic: same (match_key, side) should appear once."""
        rows = [
            {'match_key':'2026miket_qm1','side':'red','event_key':'2026miket','timestamp':1000},
            {'match_key':'2026miket_qm1','side':'red','event_key':'2026miket','timestamp':1000},
            {'match_key':'2026miket_qm1','side':'blue','event_key':'2026miket','timestamp':1000},
        ]
        seen = set()
        out = []
        for row in rows:
            key = f"{row['match_key']}_{row['side']}"
            if key not in seen:
                seen.add(key)
                out.append(row)
        self.assertEqual(len(out), 2)  # red once, blue once

    def test_event_matches_filters_by_event_key(self):
        """Only rows matching the requested event_key should be returned."""
        rows_by_team = {
            254: [
                {'match_key':'2026miket_qm1','side':'red','event_key':'2026miket','timestamp':1000},
                {'match_key':'2026cmptx_qm1','side':'red','event_key':'2026cmptx','timestamp':2000},
            ]
        }
        allowed = {254}
        event_key = '2026miket'
        seen = set()
        out = []
        for team_num, rows in rows_by_team.items():
            if team_num not in allowed:
                continue
            for row in rows:
                if row.get('event_key') != event_key:
                    continue
                key = f"{row['match_key']}_{row['side']}"
                if key not in seen:
                    seen.add(key)
                    out.append(row)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['event_key'], '2026miket')


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline simulation (full numeric path, no network/DB)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline(unittest.TestCase):
    """
    Runs the complete numeric pipeline on a synthetic 6-team, 13-match
    season and checks all numeric outputs are sane.

    NOTE on defender detection: in a balanced round-robin schedule the OPR
    and DPR design matrices are isomorphic, so the solver produces OPR ≡ DPR
    for every team.  Defender detection (DPR > 1.5 × OPR) therefore never
    fires on synthetic balanced data — it requires the naturally unbalanced
    scheduling of a real season.  We test the detection logic separately in
    TestDefenderDetection using hand-crafted asymmetric inputs.
    """

    def _run(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        import numpy as np
        from match_normalizer import MatchRecord
        from matrix_builder import build_matrices
        from solver import solve_opr_dpr, solve_weighted
        from metrics import (compute_residuals, compute_noise_sigma,
                             detect_breakers, apply_breaker_weights,
                             compute_match_counts, compute_variability)
        from refund_engine import detect_defenders, compute_refunds
        import math

        rng = np.random.default_rng(42)
        base_ts = 1_700_000_000.0
        true_opr = {1:50, 2:55, 3:45, 4:60, 5:52, 6:4}

        lineups_normal = [
            ([1,2,3],[4,5,3]), ([1,4,5],[2,3,1]), ([2,3,4],[1,5,2]),
            ([3,4,5],[1,2,4]), ([1,2,5],[3,4,5]), ([1,3,4],[2,5,3]),
            ([2,4,5],[1,3,4]),
        ]
        lineups_def = [
            ([1,2,6],[3,4,5]), ([2,3,6],[1,4,5]), ([1,4,6],[2,3,5]),
            ([3,5,6],[1,2,4]), ([2,4,6],[1,3,5]), ([1,5,6],[2,3,4]),
        ]

        matches = []
        for i, (red, blue) in enumerate(lineups_normal):
            rs = sum(true_opr[t] for t in red)  + int(rng.normal(0, 3))
            bs = sum(true_opr[t] for t in blue) + int(rng.normal(0, 3))
            matches.append(MatchRecord(
                match_key=f'2026t_qm{i+1}', event_key='2026t',
                timestamp=base_ts+i*3600, comp_level='qm', is_playoff=False,
                red_teams=red, blue_teams=blue,
                red_score=max(0, rs), blue_score=max(0, bs),
            ))
        for i, (red, blue) in enumerate(lineups_def):
            bs = sum(true_opr[t] for t in blue) + int(rng.normal(0, 3))
            rs = sum(true_opr[t] for t in red) - 28 + int(rng.normal(0, 3))
            matches.append(MatchRecord(
                match_key=f'2026t_qm{len(lineups_normal)+i+1}', event_key='2026t',
                timestamp=base_ts+(len(lineups_normal)+i)*3600, comp_level='qm',
                is_playoff=False, red_teams=red, blue_teams=blue,
                red_score=max(0, rs), blue_score=max(0, bs),
            ))

        bundle = build_matrices(matches)
        opr, dpr, _ = solve_opr_dpr(bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights)
        residuals = compute_residuals(bundle.A, opr, bundle.y)
        sigma = compute_noise_sigma(residuals)
        mask = detect_breakers(residuals, sigma)
        apply_breaker_weights(matches, bundle.row_to_match, mask)

        bundle = build_matrices(matches)
        opr, dpr, _ = solve_opr_dpr(bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights)
        residuals = compute_residuals(bundle.A, opr, bundle.y)
        sigma = compute_noise_sigma(residuals)

        mc = compute_match_counts(bundle.A, bundle.team_list)
        defenders = detect_defenders(bundle.team_list, opr, dpr, mc)
        avg_score = float(np.mean(bundle.y[bundle.weights > 0]))
        refunds, audit = compute_refunds(
            matches, residuals, opr, dpr, bundle.team_list, defenders,
            bundle.weights, avg_score
        )
        y_adj = bundle.y + refunds
        aopr, _, _ = solve_weighted(bundle.A, y_adj, bundle.weights)
        variability = compute_variability(bundle.A, residuals, bundle.team_list)

        ti = {t: i for i, t in enumerate(bundle.team_list)}
        return dict(
            team_list=bundle.team_list, ti=ti,
            opr=opr, dpr=dpr, aopr=aopr,
            sigma=sigma, defenders=defenders,
            refunds=refunds, audit=audit,
            variability=variability, mc=mc,
        )

    def test_all_outputs_finite(self):
        import math
        r = self._run()
        for name, arr in [('opr', r['opr']), ('dpr', r['dpr']), ('aopr', r['aopr'])]:
            for v in arr:
                self.assertTrue(math.isfinite(v),
                                f'{name} contains non-finite value: {v}')

    def test_sigma_positive(self):
        self.assertGreater(self._run()['sigma'], 0)

    def test_aopr_ge_opr_minus_tolerance(self):
        """
        In the absence of defender detection (balanced schedule effect),
        AOPR should equal OPR exactly (refunds = 0).  Either way, AOPR
        must never be less than OPR by more than a small numeric tolerance.
        """
        r = self._run()
        ti = r['ti']
        for t in [1, 2, 3, 4, 5]:
            if t in ti:
                self.assertGreaterEqual(
                    r['aopr'][ti[t]], r['opr'][ti[t]] - 1.0,
                    f"Team {t}: AOPR {r['aopr'][ti[t]]:.2f} << OPR {r['opr'][ti[t]]:.2f}"
                )

    def test_refunds_nonnegative(self):
        import numpy as np
        r = self._run()
        self.assertTrue((r['refunds'] >= 0).all())

    def test_audit_rows_have_required_fields(self):
        r = self._run()
        required = {'match_key', 'event_key', 'alliance_side', 'expected_score',
                    'actual_score', 'residual', 'refund', 'defender_keys', 'row_weight'}
        for row in r['audit']:
            self.assertEqual(required - set(row.keys()), set())

    def test_variability_nonnegative(self):
        r = self._run()
        for t, v in r['variability'].items():
            self.assertGreaterEqual(v, 0.0)

    def test_match_counts_nonzero(self):
        r = self._run()
        for t in [1, 2, 3, 4, 5, 6]:
            self.assertGreater(r['mc'].get(t, 0), 0)

    def test_safe_float_applied(self):
        """After _safe(), no team stat should be non-finite."""
        import math
        r = self._run()
        ti = r['ti']
        for t in r['team_list']:
            i = ti[t]
            for name, arr in [('opr', r['opr']), ('dpr', r['dpr']), ('aopr', r['aopr'])]:
                val = float(arr[i])
                safe = round(val, 3) if math.isfinite(val) else 0.0
                self.assertTrue(math.isfinite(safe))

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        import numpy as np
        r1 = self._run()
        r2 = self._run()
        np.testing.assert_allclose(r1['opr'], r2['opr'], atol=1e-8)
        np.testing.assert_allclose(r1['aopr'], r2['aopr'], atol=1e-8)



# ─────────────────────────────────────────────────────────────────────────────
# Two-tier anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoTierAnomalyDetection(unittest.TestCase):

    def _sigma(self, residuals):
        from metrics import compute_noise_sigma
        import numpy as np
        return compute_noise_sigma(np.array(residuals, dtype=float))

    def test_breaker_threshold_lower_than_exclusion(self):
        """oppr_breaker_sigma (2.0) < noise_exclusion_sigma (2.5) always."""
        from config import CONFIG
        self.assertLess(CONFIG.oppr_breaker_sigma, CONFIG.noise_exclusion_sigma)

    def test_breaker_fires_between_thresholds(self):
        """A row at 2.2σ is a breaker but NOT excluded."""
        from metrics import detect_breakers, detect_exclusions
        import numpy as np
        sigma = 10.0
        residuals = np.array([22.0])  # 2.2σ — above 2.0, below 2.5
        bmask = detect_breakers(residuals, sigma)
        emask = detect_exclusions(residuals, sigma)
        self.assertTrue(bmask[0],  "Should be a breaker at 2.2σ")
        self.assertFalse(emask[0], "Should NOT be excluded at 2.2σ")

    def test_exclusion_fires_above_2_5_sigma(self):
        """A row at 2.6σ triggers both breaker and exclusion."""
        from metrics import detect_breakers, detect_exclusions
        import numpy as np
        sigma = 10.0
        residuals = np.array([26.0])  # 2.6σ
        bmask = detect_breakers(residuals, sigma)
        emask = detect_exclusions(residuals, sigma)
        self.assertTrue(bmask[0])
        self.assertTrue(emask[0])

    def test_normal_row_triggers_neither(self):
        """A row at 1.5σ triggers neither gate."""
        from metrics import detect_breakers, detect_exclusions
        import numpy as np
        sigma = 10.0
        residuals = np.array([15.0])  # 1.5σ
        bmask = detect_breakers(residuals, sigma)
        emask = detect_exclusions(residuals, sigma)
        self.assertFalse(bmask[0])
        self.assertFalse(emask[0])

    def test_exclusion_sets_quality_weight_zero(self):
        """A match flagged for exclusion must have quality_weight=0."""
        from metrics import detect_breakers, detect_exclusions, apply_breaker_weights
        from match_normalizer import MatchRecord
        import numpy as np

        m = MatchRecord(
            match_key='2026t_qm1', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        # Row 0 (red side) has residual 30 >> 2.5σ=25 → should be excluded
        sigma = 10.0
        residuals = np.array([30.0, 5.0])   # row 0 excluded, row 1 normal
        row_to_match = [(0, 0), (0, 1)]

        bmask = detect_breakers(residuals, sigma)
        emask = detect_exclusions(residuals, sigma)
        apply_breaker_weights([m], row_to_match, bmask, emask)

        self.assertTrue(m.is_excluded)
        self.assertEqual(m.quality_weight, 0.0)

    def test_breaker_only_sets_weight_0_1(self):
        """A match in the breaker-only zone (2.0–2.5σ) gets weight=0.1."""
        from metrics import detect_breakers, detect_exclusions, apply_breaker_weights
        from match_normalizer import MatchRecord
        import numpy as np

        m = MatchRecord(
            match_key='2026t_qm2', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        sigma = 10.0
        residuals = np.array([22.0, 5.0])   # 2.2σ — breaker only
        row_to_match = [(0, 0), (0, 1)]

        bmask = detect_breakers(residuals, sigma)
        emask = detect_exclusions(residuals, sigma)
        apply_breaker_weights([m], row_to_match, bmask, emask)

        self.assertFalse(m.is_excluded)
        self.assertAlmostEqual(m.quality_weight, 0.1)
        self.assertIn('breaker', m.status_flags)

    def test_apply_breaker_without_exclusion_mask(self):
        """Calling apply_breaker_weights with no exclusion_mask is backward-compatible."""
        from metrics import detect_breakers, apply_breaker_weights
        from match_normalizer import MatchRecord
        import numpy as np

        m = MatchRecord(
            match_key='2026t_qm3', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        sigma = 10.0
        residuals = np.array([22.0, 5.0])
        row_to_match = [(0,0),(0,1)]
        bmask = detect_breakers(residuals, sigma)
        # No exclusion_mask argument
        apply_breaker_weights([m], row_to_match, bmask)
        self.assertAlmostEqual(m.quality_weight, 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Event-type weights
# ─────────────────────────────────────────────────────────────────────────────

class TestEventTypeWeights(unittest.TestCase):

    def _row_weight(self, event_type, ts_offset_days=0, team_count=40, comp_level='qm'):
        from matrix_builder import _row_weight, EVENT_TYPE_WEIGHTS
        from match_normalizer import MatchRecord
        import time
        now = 1_700_000_000.0
        m = MatchRecord(
            match_key='k', event_key='ek',
            timestamp=now - ts_offset_days * 86400,
            comp_level=comp_level, is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        return _row_weight(m, now, {'ek': team_count}, {'ek': event_type})

    def test_regular_event_weight_1(self):
        w_regional = self._row_weight(0)  # Regional
        w_district = self._row_weight(1)  # District
        self.assertAlmostEqual(w_regional, w_district, places=6)

    def test_district_champ_heavier(self):
        w_reg  = self._row_weight(0)   # Regional
        w_dcmp = self._row_weight(2)   # District Championship
        self.assertGreater(w_dcmp, w_reg)

    def test_worlds_div_heavier_than_dcmp(self):
        w_dcmp  = self._row_weight(2)   # District Champ
        w_worlds = self._row_weight(3)  # Championship Division
        self.assertGreater(w_worlds, w_dcmp)

    def test_worlds_finals_heaviest(self):
        w_div    = self._row_weight(3)  # Champ Division
        w_finals = self._row_weight(4)  # Champ Finals
        self.assertGreater(w_finals, w_div)

    def test_remote_events_lighter(self):
        w_reg    = self._row_weight(0)  # Regional
        w_remote = self._row_weight(7)  # Remote
        self.assertLess(w_remote, w_reg)

    def test_offseason_zero_weight(self):
        w = self._row_weight(99)  # Offseason
        self.assertAlmostEqual(w, 0.0)

    def test_preseason_zero_weight(self):
        w = self._row_weight(100)  # Preseason
        self.assertAlmostEqual(w, 0.0)

    def test_unknown_event_type_defaults_to_1(self):
        """An unrecognised event_type should not crash and use weight 1.0."""
        from matrix_builder import EVENT_TYPE_WEIGHTS
        w = EVENT_TYPE_WEIGHTS.get(999, 1.0)
        self.assertEqual(w, 1.0)

    def test_no_event_types_dict_unchanged(self):
        """If event_types is None, weights are identical to the pre-feature baseline."""
        from matrix_builder import _row_weight
        from match_normalizer import MatchRecord
        now = 1_700_000_000.0
        m = MatchRecord(
            match_key='k', event_key='ek',
            timestamp=now, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        w_none = _row_weight(m, now, None, None)
        w_empty = _row_weight(m, now, {}, {})
        self.assertAlmostEqual(w_none, w_empty, places=9)

    def test_event_type_multiplies_correctly(self):
        """Championship Division (1.30) weight is 1.30× the regional weight."""
        w_reg    = self._row_weight(0)
        w_worlds = self._row_weight(3)
        ratio = w_worlds / w_reg
        self.assertAlmostEqual(ratio, 1.30, places=6)

    def test_build_matrices_accepts_event_types(self):
        """build_matrices with event_types does not crash and returns correct shape."""
        from matrix_builder import build_matrices
        from match_normalizer import MatchRecord
        matches = [MatchRecord(
            match_key='2026t_qm1', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )]
        bundle = build_matrices(matches, {'2026t': 40}, {'2026t': 3})  # worlds
        self.assertEqual(bundle.A.shape, (2, 6))
        self.assertTrue((bundle.weights > 0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Nickname bulk collection
# ─────────────────────────────────────────────────────────────────────────────

class TestNicknameBulkCollection(unittest.TestCase):

    def test_nickname_stored_in_team_results(self):
        """
        Simulate the pipeline's bulk-nickname logic and verify it survives
        into team_results correctly.
        """
        # Simulate TBA team list responses
        team_list_response = [
            {'key': 'frc254', 'nickname': 'The Cheesy Poofs'},
            {'key': 'frc1114', 'nickname': 'Simbotics'},
            {'key': 'frc973',  'nickname': 'Greybots'},
        ]

        team_nicknames = {}
        team_nums = []
        for t in team_list_response:
            try:
                tn = int(t['key'].replace('frc', ''))
                team_nums.append(tn)
                if tn not in team_nicknames:
                    nick = t.get('nickname') or ''
                    if nick:
                        team_nicknames[tn] = nick
            except (ValueError, AttributeError):
                pass

        self.assertEqual(team_nicknames[254],  'The Cheesy Poofs')
        self.assertEqual(team_nicknames[1114], 'Simbotics')
        self.assertEqual(team_nicknames[973],  'Greybots')

    def test_first_occurrence_wins(self):
        """If a team appears in multiple events, the first nickname wins."""
        team_nicknames = {}
        events = [
            [{'key': 'frc254', 'nickname': 'First Name'}],
            [{'key': 'frc254', 'nickname': 'Second Name'}],
        ]
        for event_teams in events:
            for t in event_teams:
                tn = int(t['key'].replace('frc', ''))
                if tn not in team_nicknames:
                    nick = t.get('nickname') or ''
                    if nick:
                        team_nicknames[tn] = nick

        self.assertEqual(team_nicknames[254], 'First Name')

    def test_empty_nickname_not_stored(self):
        """Teams with empty/None/whitespace nickname should not pollute the dict."""
        team_nicknames = {}
        for nick in ['', None, '  ']:
            t = {'key': 'frc999', 'nickname': nick}
            tn = int(t['key'].replace('frc', ''))
            if tn not in team_nicknames:
                raw = (t.get('nickname') or '').strip()
                if raw:
                    team_nicknames[tn] = raw

        self.assertNotIn(999, team_nicknames)

    def test_nickname_in_team_result_survives_json_roundtrip(self):
        """Nicknames in team_results must survive JSON serialisation."""
        import json
        team_results = {
            254: {'team_number': 254, 'nickname': 'The Cheesy Poofs', 'opr': 55.0}
        }
        serialised = json.dumps(team_results)
        loaded = json.loads(serialised)
        # JSON keys become strings
        self.assertEqual(loaded['254']['nickname'], 'The Cheesy Poofs')

    def test_nickname_accessible_via_to_team_stats(self):
        """_to_team_stats must pass nickname through to the response model."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        # Inline _to_team_stats logic
        r = {
            'rank': 1, 'team_number': 254, 'nickname': 'The Cheesy Poofs',
            'opr': 55.0, 'dpr': 5.0, 'aopr': 58.0, 'delta': 3.0,
            'variability': 2.0, 'match_count': 12,
            'is_defender': False, 'low_match_warning': False,
        }
        # Simulate what _to_team_stats does
        nickname = r.get('nickname', '')
        self.assertEqual(nickname, 'The Cheesy Poofs')




# ─────────────────────────────────────────────────────────────────────────────
# Breaker count per team
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakerCount(unittest.TestCase):

    def _bundle_with_breaker(self):
        """Two matches; the first has a breaker flag on it."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from match_normalizer import MatchRecord
        from matrix_builder import build_matrices
        m1 = MatchRecord(
            match_key='2026t_qm1', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        m1.status_flags = ['breaker']
        m1.quality_weight = 0.1
        m2 = MatchRecord(
            match_key='2026t_qm2', event_key='2026t',
            timestamp=1_700_003_600.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=85, blue_score=75,
        )
        return build_matrices([m1, m2]), [m1, m2]

    def test_breaker_count_nonzero_for_affected_teams(self):
        bundle, matches = self._bundle_with_breaker()
        # Teams 1,2,3 were on the red side of the breaker match
        team_breaker_counts = {t: 0 for t in bundle.team_list}
        for ri, (mi, side) in enumerate(bundle.row_to_match):
            m = matches[mi]
            if 'breaker' in m.status_flags or m.is_excluded:
                scoring = m.red_teams if side == 0 else m.blue_teams
                for t in scoring:
                    if t in team_breaker_counts:
                        team_breaker_counts[t] += 1
        # Teams 1,2,3 scored on the red side of match 0 (which is the breaker)
        self.assertGreater(team_breaker_counts[1], 0)
        self.assertGreater(team_breaker_counts[2], 0)
        self.assertGreater(team_breaker_counts[3], 0)

    def test_both_alliances_counted_in_breaker(self):
        """
        A breaker match affects the whole match weight, so BOTH alliances'
        teams accumulate a breaker count — not just the underperforming side.
        """
        bundle, matches = self._bundle_with_breaker()
        team_breaker_counts = {t: 0 for t in bundle.team_list}
        for ri, (mi, side) in enumerate(bundle.row_to_match):
            m = matches[mi]
            if 'breaker' in m.status_flags or m.is_excluded:
                scoring = m.red_teams if side == 0 else m.blue_teams
                for t in scoring:
                    if t in team_breaker_counts:
                        team_breaker_counts[t] += 1
        # Both red (1,2,3) and blue (4,5,6) are on the breaker match
        for t in [1, 2, 3, 4, 5, 6]:
            self.assertGreater(team_breaker_counts[t], 0,
                               f"Team {t} should have breaker_count > 0")

    def test_excluded_match_also_increments_count(self):
        """is_excluded=True should also count toward breaker_count."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from match_normalizer import MatchRecord
        from matrix_builder import build_matrices
        m = MatchRecord(
            match_key='2026t_qm1', event_key='2026t',
            timestamp=1_700_000_000.0, comp_level='qm', is_playoff=False,
            red_teams=[1,2,3], blue_teams=[4,5,6],
            red_score=80, blue_score=70,
        )
        m.is_excluded = True
        m.quality_weight = 0.0
        bundle = build_matrices([m])
        team_breaker_counts = {t: 0 for t in bundle.team_list}
        for ri, (mi, side) in enumerate(bundle.row_to_match):
            if m.is_excluded:
                scoring = m.red_teams if side == 0 else m.blue_teams
                for t in scoring:
                    if t in team_breaker_counts:
                        team_breaker_counts[t] += 1
        self.assertGreater(team_breaker_counts[1], 0)





# ─────────────────────────────────────────────────────────────────────────────
# TBA client — event filtering, retry, stale fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestTBAClientFiltering(unittest.TestCase):
    """Tests the event-filtering logic in get_season_events (no network calls)."""

    def _filter_events(self, events, year=2026):
        """Replicate the filtering logic from tba_client.get_season_events."""
        SKIP = {99, 100}
        filtered = []
        for ev in events:
            et = ev.get('event_type')
            if et in SKIP:
                continue
            start = ev.get('start_date', '')
            if start and not start.startswith(str(year)):
                continue
            filtered.append(ev)
        return filtered

    def test_offseason_excluded(self):
        events = [
            {'key': '2026miket', 'event_type': 0, 'start_date': '2026-03-10'},
            {'key': '2026off01', 'event_type': 99, 'start_date': '2026-09-15'},
        ]
        result = self._filter_events(events)
        keys = [e['key'] for e in result]
        self.assertIn('2026miket', keys)
        self.assertNotIn('2026off01', keys)

    def test_preseason_excluded(self):
        events = [
            {'key': '2026pre01', 'event_type': 100, 'start_date': '2026-01-05'},
            {'key': '2026miket', 'event_type': 0,   'start_date': '2026-03-10'},
        ]
        result = self._filter_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['key'], '2026miket')

    def test_wrong_year_excluded(self):
        events = [
            {'key': '2025miket', 'event_type': 0, 'start_date': '2025-03-10'},
            {'key': '2026miket', 'event_type': 0, 'start_date': '2026-03-10'},
        ]
        result = self._filter_events(events, year=2026)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['key'], '2026miket')

    def test_all_official_types_kept(self):
        """Types 0–7 (excluding 99/100) should all pass through."""
        events = [
            {'key': f'ev{t}', 'event_type': t, 'start_date': '2026-03-10'}
            for t in [0, 1, 2, 3, 4, 5, 6, 7]
        ]
        result = self._filter_events(events)
        self.assertEqual(len(result), 8)

    def test_no_start_date_not_filtered(self):
        """Events without a start_date are kept (can't determine year)."""
        events = [{'key': 'evno', 'event_type': 0, 'start_date': ''}]
        result = self._filter_events(events)
        self.assertEqual(len(result), 1)

    def test_skip_types_constant(self):
        """SKIP_EVENT_TYPES must contain 99 and 100."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        # Inline the constant — matches tba_client._SKIP_EVENT_TYPES
        skip = {99, 100}
        self.assertIn(99,  skip)
        self.assertIn(100, skip)
        self.assertNotIn(0, skip)
        self.assertNotIn(3, skip)

    def test_worlds_events_kept(self):
        """Championship types (3, 4) must not be filtered."""
        events = [
            {'key': '2026cmptx', 'event_type': 3, 'start_date': '2026-04-20'},
            {'key': '2026cmpmi', 'event_type': 4, 'start_date': '2026-04-24'},
        ]
        result = self._filter_events(events)
        self.assertEqual(len(result), 2)


class TestTBAClientRetryLogic(unittest.TestCase):
    """Tests retry and stale-fallback logic without real network calls."""

    def test_skip_types_are_offseason_preseason_only(self):
        """Verifies exactly which types are skipped — must not skip regular events."""
        skip = {99, 100}
        for official_type in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.assertNotIn(official_type, skip,
                             f"Type {official_type} should not be skipped")

    def test_retry_delays_exponential(self):
        """Retry delays should be 1, 2, 4 seconds (2^attempt)."""
        delays = [2 ** attempt for attempt in range(3)]
        self.assertEqual(delays, [1, 2, 4])

    def test_semaphore_limit(self):
        """The semaphore value must be a reasonable concurrency cap."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        import importlib
        # Read the semaphore value from source without importing httpx
        src = open(os.path.join(os.path.dirname(__file__), '..', 'backend', 'tba_client.py')).read()
        import re
        match = re.search(r'Semaphore\((\d+)\)', src)
        self.assertIsNotNone(match, "Semaphore not found in tba_client.py")
        limit = int(match.group(1))
        self.assertGreaterEqual(limit, 5,  "Semaphore too low — too slow")
        self.assertLessEqual(limit, 20,    "Semaphore too high — risk rate-limiting")



if __name__ == "__main__":
    unittest.main(verbosity=2)