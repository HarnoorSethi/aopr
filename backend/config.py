import dataclasses
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AOPRConfig:
    """Tunable parameters for the AOPR solver. Used for caching keys and metadata."""
    ridge_lambda: float = 0.0   # 0.0 = adaptive; positive value overrides auto-damp
    noise_sigma: float = 2.5    # residual exclusion threshold (in σ units)
    min_matches: int = 6        # minimum matches before a team is ranked

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SeasonConfig:
    season_year: int = field(default_factory=lambda: datetime.now().year)
    refresh_interval_minutes: int = 10
    time_decay_half_life_days: float = 21.0
    min_matches_to_rank: int = 6
    defender_threshold_multiplier: float = 1.5
    noise_exclusion_sigma: float = 2.5
    oppr_breaker_sigma: float = 2.0
    dpr_weight_power: float = 1.5   # raise existing weights to this power for DPR/QDPR solves (>1 = more recency)
    tba_base_url: str = "https://www.thebluealliance.com/api/v3"
    tba_auth_key: str = "jFudwoGL3FcSrLXIACgqzo0raHKl2N1A2G0Mkk23967k3XusqXL1bVCQYoz813Bb"
    db_path: str = field(default_factory=lambda: os.environ.get("DB_PATH", "aopr.db"))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    aopr: AOPRConfig = field(default_factory=AOPRConfig)

    def __post_init__(self) -> None:
        # Keep AOPRConfig in sync with the primary fields so both paths work.
        self.aopr.noise_sigma = self.noise_exclusion_sigma
        self.aopr.min_matches = self.min_matches_to_rank


CONFIG = SeasonConfig()
