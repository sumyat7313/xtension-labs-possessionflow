"""Exact Viterbi decoding with a minimum duration for internal state segments."""

import numpy as np


def decode(probabilities, fps, config, team_switch_costs=None, *, early_exit_allowed=None,
           min_short_team_seconds=.16, early_exit_penalty=1.):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not len(probabilities):
        raise ValueError("Decoder expects a nonempty n by 3 probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Decoder probabilities must be finite, nonnegative, and sum to one")
    if team_switch_costs is not None:
        team_switch_costs = np.asarray(team_switch_costs, dtype=float)
        if (team_switch_costs.shape != (len(probabilities),) or not np.isfinite(team_switch_costs).all()
                or (team_switch_costs < 0).any()):
            raise ValueError("Team switch costs must be a finite nonnegative per-frame vector")
    if early_exit_allowed is not None:
        early_exit_allowed = np.asarray(early_exit_allowed)
        if early_exit_allowed.shape != (len(probabilities),) or early_exit_allowed.dtype != np.bool_:
            raise ValueError("Early exits must be a per-frame boolean vector")
        if (isinstance(min_short_team_seconds, bool) or not np.isfinite(min_short_team_seconds)
                or not 0 < min_short_team_seconds <= config.min_team_seconds
                or isinstance(early_exit_penalty, bool) or not np.isfinite(early_exit_penalty) or early_exit_penalty < 0):
            raise ValueError("Invalid short-possession floor or early-exit penalty")
    durations = np.maximum(1, np.rint(np.array([config.min_team_seconds, config.min_team_seconds,
                                              config.min_dead_seconds]) * fps).astype(int))
    starts = np.r_[0, np.cumsum(durations)[:-1]]
    ends = np.cumsum(durations) - 1
    size = int(durations.sum())
    if size > 255:
        raise ValueError("Minimum durations create more than 255 expanded states")
    owners = np.repeat(np.arange(3, dtype=np.uint8), durations)
    switch = np.array([[np.inf, config.team_switch_cost, config.stop_cost],
                       [config.team_switch_cost, np.inf, config.stop_cost],
                       [config.restart_cost, config.restart_cost, np.inf]])
    emission = -np.log(np.maximum(probabilities, 1e-8)) * config.effective_observations_per_second / fps
    previous = np.full(size, np.inf)
    # The first state may already be in progress; the last may be truncated.
    previous[ends] = emission[0]
    back = np.zeros((len(probabilities), size), dtype=np.uint8)
    age_destinations = np.array([j for j in range(size) if j not in starts], dtype=int)
    short_duration = max(1, int(np.ceil(min_short_team_seconds * fps))) if early_exit_allowed is not None else 0
    for frame in range(1, len(probabilities)):
        if team_switch_costs is not None:
            switch[0, 1] = switch[1, 0] = team_switch_costs[frame]
        current = np.full(size, np.inf)
        current[age_destinations] = previous[age_destinations - 1] + emission[frame, owners[age_destinations]]
        back[frame, age_destinations] = age_destinations - 1
        for state in range(3):
            end = ends[state]
            stay = previous[end] + emission[frame, state]
            if stay <= current[end]:
                current[end] = stay
                back[frame, end] = end
            transitions = previous[ends] + switch[:, state]
            winner = int(np.argmin(transitions))
            value = transitions[winner] + emission[frame, state]
            if value < current[starts[state]]:
                current[starts[state]] = value
                back[frame, starts[state]] = ends[winner]
            if state < 2 and early_exit_allowed is not None and early_exit_allowed[frame]:
                # Only H<->A may leave a short state early. D entry/exit retains
                # the original duration constraints and penalties.
                source = 1 - state
                eligible = np.arange(starts[source] + short_duration - 1, ends[source])
                if len(eligible):
                    origin = int(eligible[np.argmin(previous[eligible])])
                    value = previous[origin] + switch[source, state] + early_exit_penalty + emission[frame, state]
                    if value < current[starts[state]]:
                        current[starts[state]] = value
                        back[frame, starts[state]] = origin
        # Subtract a shared offset to bound accumulated floating point error.
        current -= np.min(current)
        previous = current
    result = np.empty(len(probabilities), dtype=np.uint8)
    state = int(np.argmin(previous))
    for frame in range(len(probabilities) - 1, -1, -1):
        result[frame] = owners[state]
        state = int(back[frame, state])
    return result


def observation_probabilities(features, tags, config):
    log_odds = np.clip(features.team_log_odds + tags.team_log_odds, -8, 8)
    home = 1 / (1 + np.exp(-log_odds))
    dead_logit = (
        config.dead_logit_bias
        + config.outside_dead_weight * features.outside
        + config.still_dead_weight * features.stationary * features.still_players
        + config.restart_dead_weight * tags.dead
        + config.missing_still_weight * features.missing * features.still_players
        - config.phase_live_weight * tags.live
        - config.motion_live_weight * np.maximum(features.moving, tags.restart_live)
    )
    dead = 1 / (1 + np.exp(-np.clip(dead_logit, -8, 8)))
    return np.column_stack([(1 - dead) * home, (1 - dead) * (1 - home), dead])
