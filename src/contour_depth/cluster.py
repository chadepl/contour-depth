import numpy as np
from enum import Enum


class Depth(Enum):
    EpsilonInclusionDepth = 1
    InclusionDepth = 2
    ContourBandDepth = 3


class Metric(Enum):
    Depth = 1
    RelativeDepth = 2


def cluster_inclusion_matrix(
        masks,
        num_clusters,
        depth=Depth.EpsilonInclusionDepth,
        metric=Metric.Depth,
        kmeans_num_attempts=5,
        kmeans_max_iterations=10,
        kmeans_random_seed=42):
    masks = np.array(masks, dtype=np.float32)
    num_masks = masks.shape[0]
    # or depth == Depth.EpsilonContourBandDepth:
    if depth == Depth.EpsilonInclusionDepth:
        inclusion_matrix = compute_epsilon_inclusion_matrix(masks)
        # Required for feature parity with the O(N) version of eID.
        np.fill_diagonal(inclusion_matrix, 1)
    else:
        inclusion_matrix = compute_inclusion_matrix(masks)

    if kmeans_random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(kmeans_random_seed)

    def check_valid_assignment(assignment, num_clusters):
        for c in range(num_clusters):
            if np.sum(assignment == c) < 3:
                return False
        return True

    best_depth_sum = -np.inf
    best_cluster_assignment = None
    for _ in range(kmeans_num_attempts):
        cluster_assignment = rng.integers(
            low=0, high=num_clusters, size=num_masks)
        for _ in range(kmeans_max_iterations):
            depth_in_cluster = np.empty(
                (num_clusters, num_masks), dtype=np.float32)
            for c in range(num_clusters):
                j_in_cluster = cluster_assignment == c

                N = np.sum(j_in_cluster)
                if depth == Depth.ContourBandDepth or depth == Depth.InclusionDepth or depth == Depth.EpsilonInclusionDepth:
                    N_a = np.sum(inclusion_matrix[:, j_in_cluster], axis=1)
                    N_b = np.sum(inclusion_matrix.T[:, j_in_cluster], axis=1)

                    if depth == Depth.ContourBandDepth:
                        # We need to normalize the depth such that it is  not dependent on the number of contours in the cluster.
                        # If the contour is already in the cluster then N_a and N_b range from 0 to N-1
                        # If the contour is *not* in the cluster then N_a and
                        # N_b range from 0 to N
                        N_ab_range = N - j_in_cluster
                        depth_in_cluster[c] = (
                            N_a * N_b) / (N_ab_range * N_ab_range)
                    else:  # ID / eID
                        depth_in_cluster[c] = np.minimum(N_a, N_b) / N
                else:
                    assert False, f"Unsupported depth {depth}"

            if metric == Metric.Depth:
                metric_values = depth_in_cluster
            elif metric == Metric.RelativeDepth:
                red = np.empty(depth_in_cluster.shape, dtype=np.float32)
                for c in range(num_clusters):
                    # Compute the max value exluding the current cluster.
                    # There is a more efficient, but slightly dirtier,
                    # solution.
                    depth_between = np.max(
                        np.roll(depth_in_cluster, -c, axis=0)[1:, :], axis=0)
                    depth_within = depth_in_cluster[c, :]
                    red[c, :] = depth_within - depth_between
                metric_values = red
            else:
                assert False, f"Unsupported metric {metric}"

            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            if not check_valid_assignment(
                    cluster_assignment, num_clusters) or np.all(
                    cluster_assignment == old_cluster_assignment):
                break

            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment


def cluster_optimized_eid(
        masks,
        num_clusters,
        metric=Metric.Depth,
        kmeans_num_attempts=5,
        kmeans_max_iterations=10,
        kmeans_random_seed=42):
    masks = np.array(masks, dtype=np.float32)
    num_masks, height, width = masks.shape
    neg_masks = 1 - masks
    areas = np.sum(masks, axis=(1, 2))
    inv_areas = 1 / areas

    if kmeans_random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(kmeans_random_seed)

    best_depth_sum = -np.inf
    best_cluster_assignment = None
    for _ in range(kmeans_num_attempts):
        cluster_assignment = rng.integers(
            low=0, high=num_clusters, size=num_masks)
        for _ in range(kmeans_max_iterations):
            precompute_in = np.empty(
                (num_clusters, height, width), dtype=np.float32)
            precompute_out = np.empty(
                (num_clusters, height, width), dtype=np.float32)

            for c in range(num_clusters):
                j_in_cluster = cluster_assignment == c
                selected_masks = masks[j_in_cluster]
                selected_areas = areas[j_in_cluster]
                selected_inv_masks = neg_masks[j_in_cluster]

                precompute_in[c] = np.sum(selected_inv_masks, axis=0)
                precompute_out[c] = np.sum(
                    (selected_masks.T / selected_areas.T).T, axis=0)

            depth_in_cluster = np.empty(
                (num_clusters, num_masks), dtype=np.float32)
            empty_cluster = False
            for c in range(num_clusters):
                N = np.sum(cluster_assignment == c)
                if N == 0:
                    empty_cluster = True
                    break
                IN_in = N - inv_areas * \
                    np.sum(masks * precompute_in[c], axis=(1, 2))
                IN_out = N - np.sum(neg_masks * precompute_out[c], axis=(1, 2))
                depth_in_cluster[c] = np.minimum(IN_in, IN_out) / N
            if empty_cluster:
                break

            if metric == Metric.Depth:
                metric_values = depth_in_cluster
            elif metric == Metric.RelativeDepth:
                red = np.empty(depth_in_cluster.shape, dtype=np.float32)
                for c in range(num_clusters):
                    # Compute the max value exluding the current cluster.
                    # There is a more efficient, but slightly dirtier,
                    # solution.
                    depth_between = np.max(
                        np.roll(depth_in_cluster, -c, axis=0)[1:, :], axis=0)
                    depth_within = depth_in_cluster[c, :]
                    red[c, :] = depth_within - depth_between
                metric_values = red
            else:
                assert False, f"Unsupported metric {metric}"

            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            if np.all(cluster_assignment == old_cluster_assignment):
                break
            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment


def compute_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous.

    Parameters
    ----------
    masks : _type_
        _description_
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all(
            (masks & masks[i]) == masks[i], axis=(1, 2))
        inclusion_mat[i, i] = 0
    return inclusion_mat


def compute_epsilon_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous.

    Parameters
    ----------
    masks : _type_
        _description_
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    for i in range(num_masks):
        inclusion_mat[i, :] = 1 - \
            np.sum(inv_masks & masks[i], axis=(1, 2)) / np.sum(masks[i])
        inclusion_mat[i, i] = 0
    return inclusion_mat
