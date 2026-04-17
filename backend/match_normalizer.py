from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_COMP_LEVELS = {"qm", "ef", "qf", "sf", "f"}

STAGE_WEIGHTS: Dict[str, float] = {
    "qm": 1.00,
    "ef": 1.20,
    "qf": 1.30,
    "sf": 1.35,
    "f":  1.50,
}


@dataclass
class MatchRecord:
    match_key: str
    event_key: str
    timestamp: float
    comp_level: str
    is_playoff: bool
    red_teams: List[int]
    blue_teams: List[int]
    red_score: int
    blue_score: int
    status_flags: List[str] = field(default_factory=list)
    is_replay: bool = False
    is_excluded: bool = False
    quality_weight: float = 1.0  # adjusted later by breaker/anomaly detection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_matches(raw_matches: List[dict], event_key: str) -> List[MatchRecord]:
    """
    Convert raw TBA match dicts into clean MatchRecord objects.

    Rules applied:
      - Skip non-FRC comp levels
      - Skip unplayed / invalid-score matches
      - For replays: keep only the latest version of a given slot
      - Exclude surrogate/DQ teams (they distort the model)
    """
    # Group by canonical slot: (comp_level, set_number, match_number)
    slots: Dict[Tuple, List[dict]] = {}
    for m in raw_matches:
        cl = m.get("comp_level")
        if cl not in VALID_COMP_LEVELS:
            continue
        slot = (cl, m.get("set_number", 1), m.get("match_number", 0))
        slots.setdefault(slot, []).append(m)

    records: List[MatchRecord] = []
    for slot, candidates in slots.items():
        valid = [m for m in candidates if _has_valid_scores(m)]
        if not valid:
            continue

        # Latest played match wins (handles replays deterministically)
        valid.sort(
            key=lambda m: m.get("actual_time") or m.get("post_result_time") or m.get("predicted_time") or 0,
            reverse=True,
        )
        m = valid[0]
        is_replay = len(valid) > 1

        red_teams, blue_teams = _parse_alliance_teams(m)
        if not red_teams or not blue_teams:
            logger.debug("Skipping %s: could not parse teams", m.get("key"))
            continue

        alliances = m.get("alliances", {})
        red_score = alliances.get("red", {}).get("score", -1)
        blue_score = alliances.get("blue", {}).get("score", -1)

        comp_level = m["comp_level"]
        ts = float(
            m.get("actual_time") or m.get("post_result_time") or m.get("predicted_time") or 0
        )

        flags: List[str] = []
        if is_replay:
            flags.append("replay")

        records.append(
            MatchRecord(
                match_key=m["key"],
                event_key=event_key,
                timestamp=ts,
                comp_level=comp_level,
                is_playoff=comp_level != "qm",
                red_teams=red_teams,
                blue_teams=blue_teams,
                red_score=int(red_score),
                blue_score=int(blue_score),
                status_flags=flags,
                is_replay=is_replay,
            )
        )

    logger.info("Normalized %d matches from %d slots for %s", len(records), len(slots), event_key)
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_valid_scores(m: dict) -> bool:
    alliances = m.get("alliances", {})
    red_score = alliances.get("red", {}).get("score", -1)
    blue_score = alliances.get("blue", {}).get("score", -1)
    return red_score is not None and blue_score is not None and red_score >= 0 and blue_score >= 0


def _parse_alliance_teams(m: dict) -> Tuple[List[int], List[int]]:
    alliances = m.get("alliances", {})
    red_info = alliances.get("red", {})
    blue_info = alliances.get("blue", {})

    red_surrogates = set(red_info.get("surrogate_team_keys", []))
    blue_surrogates = set(blue_info.get("surrogate_team_keys", []))
    red_dq = set(red_info.get("dq_team_keys", []))
    blue_dq = set(blue_info.get("dq_team_keys", []))

    red_teams = _extract_team_numbers(
        red_info.get("team_keys", []),
        excluded=red_surrogates | red_dq,
    )
    blue_teams = _extract_team_numbers(
        blue_info.get("team_keys", []),
        excluded=blue_surrogates | blue_dq,
    )

    return red_teams, blue_teams


def _extract_team_numbers(team_keys: List[str], excluded: set) -> List[int]:
    result = []
    for k in team_keys:
        if k in excluded:
            continue
        try:
            result.append(int(k.replace("frc", "")))
        except (ValueError, AttributeError):
            pass
    return result
