"""Exact Viterbi decoding with a minimum duration for internal state segments."""

import numpy as np


def decode(probabilities, fps, config):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not len(probabilities):
        raise ValueError("Decoder expects a nonempty n by 3 probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Decoder probabilities must be finite, nonnegative, and sum to one")
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
    for frame in range(1, len(probabilities)):
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
