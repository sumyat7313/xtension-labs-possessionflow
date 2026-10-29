"""Physical and temporal defaults, fixed without using the public labels."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    max_ball_speed_mps: float = 45.0
    interpolation_seconds: float = 0.32
    interpolated_weight: float = 0.45
    control_radius_m: float = 1.5
    control_decay_m: float = 2.0
    distance_margin_m: float = 1.5
    fast_ball_mps: float = 14.0
    stationary_ball_mps: float = 0.8
    still_players_mps: float = 1.0
    outside_margin_m: float = 0.4
    evidence_window_seconds: float = 0.6
    tag_taper_seconds: float = 0.8
    tag_log_odds: float = 2.0
    restart_context_seconds: float = 3.0
    restart_before_seconds: float = 0.6
    restart_after_seconds: float = 0.4
    restart_stationary_mps: float = 1.2
    restart_moving_mps: float = 2.0
    min_team_seconds: float = 0.4
    min_dead_seconds: float = 0.6
    effective_observations_per_second: float = 5.0
    team_switch_cost: float = 4.0
    stop_cost: float = 3.0
    restart_cost: float = 2.8
    dead_logit_bias: float = -2.0
    outside_dead_weight: float = 5.5
    still_dead_weight: float = 2.4
    restart_dead_weight: float = 4.0
    missing_still_weight: float = 1.5
    phase_live_weight: float = 1.2
    motion_live_weight: float = 1.5

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if field.name != "dead_logit_bias" and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.interpolated_weight > 1:
            raise ValueError("interpolated_weight must be <= 1")
        return self

    def as_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        return Config().validate()
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return Config(**values).validate()
