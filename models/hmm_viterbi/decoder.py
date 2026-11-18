import numpy as np


EARTH_RADIUS_M = 6371008.8


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def transition_matrix(prev, curr, arcs, gps_distance, params):
    prev_ids = prev["arc_id"].to_numpy(dtype=np.int64)
    curr_ids = curr["arc_id"].to_numpy(dtype=np.int64)
    prev_t = prev["proj_t"].to_numpy(dtype=float)
    curr_t = curr["proj_t"].to_numpy(dtype=float)
    prev_len = prev["arc_len"].to_numpy(dtype=float)
    curr_len = curr["arc_len"].to_numpy(dtype=float)

    same = prev_ids[:, None] == curr_ids[None, :]
    direct = arcs[prev_ids, 1][:, None] == arcs[curr_ids, 0][None, :]
    same_distance = (curr_t[None, :] - prev_t[:, None]) * prev_len[:, None]
    direct_distance = (
        (1.0 - prev_t[:, None]) * prev_len[:, None]
        + curr_t[None, :] * curr_len[None, :]
    )

    connected = same | direct
    network_distance = np.where(same, same_distance, direct_distance)
    error = np.abs(network_distance - gps_distance)

    score = -params["weight"] * np.minimum(
        error / params["sigma_m"], params["error_cap"]
    )
    score[~connected] = -params["disconnected_penalty"]

    backwards = same & (
        same_distance < -params["backward_tolerance_m"]
    )
    score[backwards] -= params["backward_penalty"]
    return score


def decode_block(block, arcs, params):
    dp = block[0]["_emission"].to_numpy(dtype=float).copy()
    backpointers = []

    for i in range(1, len(block)):
        prev, curr = block[i - 1], block[i]
        gps_distance = haversine(
            float(prev["_lon"].iloc[0]),
            float(prev["_lat"].iloc[0]),
            float(curr["_lon"].iloc[0]),
            float(curr["_lat"].iloc[0])
        )
        trans = transition_matrix(prev, curr, arcs, gps_distance, params)
        scores = dp[:, None] + trans
        best_previous = np.argmax(scores, axis=0)
        dp = scores[best_previous, np.arange(scores.shape[1])]
        dp += curr["_emission"].to_numpy(dtype=float)
        backpointers.append(best_previous)

    state = int(np.argmax(dp))
    states = [state]
    for pointer in reversed(backpointers):
        state = int(pointer[state])
        states.append(state)
    states.reverse()

    return [
        int(block[i].iloc[state]["arc_id"])
        for i, state in enumerate(states)
    ]


def decode_trip(points, candidates, arcs, probabilities, params):
    cands = candidates.copy()
    cands["_prob"] = np.asarray(probabilities, dtype=float)

    # Index candidate groups once instead of filtering the full table per point.
    candidate_groups = {
        int(k): g.reset_index(drop=True)
        for k, g in cands.groupby("point_idx", sort=False)
    }

    predictions = {}

    for _, segment_points in points.groupby("segment", sort=False):
        segment_points = segment_points.sort_values("point_idx")
        blocks = []
        current_block = []
        previous_idx = None

        for point in segment_points.itertuples(index=False):
            point_idx = int(point.point_idx)

            if previous_idx is not None and point_idx != previous_idx + 1:
                if current_block:
                    blocks.append(current_block)
                current_block = []

            previous_idx = point_idx
            point_cands = candidate_groups.get(point_idx)

            if point_cands is None or point_cands.empty:
                if current_block:
                    blocks.append(current_block)
                current_block = []
                continue

            point_cands = point_cands.copy()
            probs = np.clip(
                point_cands["_prob"].to_numpy(dtype=float), 1e-12, 1.0
            )
            probs /= probs.sum()
            point_cands["_emission"] = np.log(probs)
            point_cands["_lon"] = float(point.lon)
            point_cands["_lat"] = float(point.lat)
            current_block.append(point_cands)

        if current_block:
            blocks.append(current_block)

        for block in blocks:
            decoded = decode_block(block, arcs, params)
            for point_cands, arc_id in zip(block, decoded):
                predictions[int(point_cands.iloc[0]["point_idx"])] = arc_id

    return predictions


def classifier_predictions(candidates, probabilities):
    cands = candidates.copy()
    cands["_prob"] = np.asarray(probabilities, dtype=float)
    indices = cands.groupby("point_idx")["_prob"].idxmax()
    selected = cands.loc[indices]
    return dict(zip(
        selected["point_idx"].astype(int),
        selected["arc_id"].astype(int)
    ))
