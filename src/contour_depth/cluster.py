import numpy as np
from enum import Enum

from .depth import Depth, compute_inclusion_matrix, compute_epsilon_inclusion_matrix, _get_agg_axes


class Metric(Enum):
    Depth = 1
    RelativeDepth = 2


def __compute_depth_in_cluster(masks:list[np.array], cluster_assignment:np.array, num_clusters:int, inclusion_matrix:np.array, depth:Depth) -> np.array:
    """Computes the depth of each contour with respect to each cluster.

    Parameters
    ----------
    masks : list[np.array]
        List of binary masks representing the contours.
    cluster_assignment : np.array
        Array with membership labels of each contour in the ensemble.
    num_clusters : int
        Number of unique labels in the clustering.
    inclusion_matrix : np.array
        Inclusion matrix for the input ensemble(`masks`)
    depth : Depth
        Choice of depth notion (see the Depth enum for available options). 

    Returns
    -------
    np.array
        Array of shape num_clusters x num_masks with the depth of each contour with respect to each cluster.
    """
    num_masks = len(masks)
    depth_in_cluster = np.empty((num_clusters, num_masks), dtype=np.float32)
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
                depth_in_cluster[c] = (N_a * N_b) / (N_ab_range * N_ab_range)
            else:  # ID / eID
                depth_in_cluster[c] = np.minimum(N_a, N_b) / N
        else:
            assert False, f"Unsupported depth {depth}"
    return depth_in_cluster


def compute_depth_in_cluster(masks:list[np.array], cluster_assignment:np.array, num_clusters:int, depth:Depth) -> np.array:
    """Computes the depth of each contour with respect to each cluster.

    Parameters
    ----------
    masks : list[np.array]
        List of binary masks representing the contours.
    cluster_assignment : np.array
        Array with membership labels of each contour in the ensemble.
    num_clusters : int
        Number of unique labels in the clustering.
    depth : Depth
        Choice of depth notion (see the Depth enum for available options). 

    Returns
    -------
    np.array
        Array of shape num_clusters x num_masks with the depth of each contour with respect to each cluster.
    """
    assert(depth != Depth.EpsilonContourBandDepth)
    if depth == Depth.ContourBandDepth or Depth == Depth.InclusionDepth:
        inclusion_matrix = compute_inclusion_matrix(masks)
    else:
        inclusion_matrix = compute_epsilon_inclusion_matrix(masks)
    return __compute_depth_in_cluster(masks, cluster_assignment, num_clusters, inclusion_matrix, depth)


def compute_relative_depth(depth_in_cluster:np.array, num_clusters:int) -> np.array:
    """Computes relative depths (depth within - depth between) of a contour ensemble.

    Parameters
    ----------
    depth_in_cluster : np.array
        Array of shape num_clusters x num_masks with the depth of each contour with respect to each cluster.
    num_clusters : int
        Number of unique labels in the clustering.

    Returns
    -------
    np.array
        Array containing the ReD of each contour.
    """
    red = np.empty(depth_in_cluster.shape, dtype=np.float32)
    for c in range(num_clusters):
        # Compute the max value exluding the current cluster.
        # There is a more efficient, but slightly dirtier,
        # solution.
        depth_between = np.max(
            np.roll(depth_in_cluster, -c, axis=0)[1:, :], axis=0)
        depth_within = depth_in_cluster[c, :]
        red[c, :] = depth_within - depth_between
    return red


def cluster_inclusion_matrix(
        masks:list[np.array],
        num_clusters:int,
        depth:Depth=Depth.EpsilonInclusionDepth,
        metric:Metric=Metric.Depth,
        kmeans_num_attempts:int=5,
        kmeans_max_iterations:int=10,
        kmeans_random_seed:int=42) -> np.array:
    """Clusters contour ensemble based on inclusion matrix.

    Parameters
    ----------
    masks : list[np.array]
        List of binary masks representing the contours.
    num_clusters : int
        Number of unique labels in the clustering.
    depth : Depth, optional
        Choice of depth notion (see the Depth enum for available options). 
    metric : Metric, optional
        Choice of metric to use for clustering.
    kmeans_num_attempts : int, optional
        Number of attempts for the clustering algorithm, by default 5.
    kmeans_max_iterations : int, optional
        Max number of iterations for the clustering algorithm, by default 10.
    kmeans_random_seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    np.array
        Clustering of the contour ensemble.
    """
    masks = np.array(masks, dtype=np.float32)
    num_masks = masks.shape[0]
    # or depth == Depth.EpsilonContourBandDepth:
    assert(depth != Depth.EpsilonContourBandDepth)
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
            depth_in_cluster = __compute_depth_in_cluster(masks, cluster_assignment, num_clusters, inclusion_matrix, depth)

            if metric == Metric.Depth:
                metric_values = depth_in_cluster
            elif metric == Metric.RelativeDepth:
                metric_values = compute_relative_depth(depth_in_cluster, num_clusters)
            else:
                assert False, f"Unsupported metric {metric}"

            old_cluster_assignment = cluster_assignment
            cluster_assignment = np.argmax(metric_values, axis=0)
            if not check_valid_assignment(cluster_assignment, num_clusters) or np.all(cluster_assignment == old_cluster_assignment):
                break

            depth_sum = np.sum(np.choose(cluster_assignment, metric_values))
            if depth_sum > best_depth_sum:
                best_cluster_assignment = cluster_assignment
                best_depth_sum = depth_sum

    return best_cluster_assignment


def cluster_optimized_eid(
        masks:list[np.array],
        num_clusters:int,
        metric:Metric=Metric.Depth,
        kmeans_num_attempts:int=5,
        kmeans_max_iterations:int=10,
        kmeans_random_seed:int=42) -> np.array:
    """Clusters contour ensemble. Optimized for the epsilon Inclusion Depth.

    Parameters
    ----------
    masks : list[np.array]
        List of binary masks representing the contours.
    num_clusters : int
        Number of unique labels in the clustering.
    depth : Depth, optional
        Choice of depth notion (see the Depth enum for available options). 
    metric : Metric, optional
        Choice of metric to use for clustering.
    kmeans_num_attempts : int, optional
        Number of attempts for the clustering algorithm, by default 5.
    kmeans_max_iterations : int, optional
        Max number of iterations for the clustering algorithm, by default 10.
    kmeans_random_seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    np.array
        Clustering of the contour ensemble.
    """
    masks = np.array(masks, dtype=np.float32)
    agg_axes = _get_agg_axes(masks)
    if len(agg_axes) == 2:
        num_masks, height, width = masks.shape
        arr_size = [height, width]
    elif len(agg_axes) == 3:
        num_masks, height, width, depth = masks.shape
        arr_size = [height, width, depth]

    neg_masks = 1 - masks
    areas = np.sum(masks, axis=agg_axes)
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
                [num_clusters, ] + arr_size, dtype=np.float32)
            precompute_out = np.empty(
                [num_clusters, ] + arr_size, dtype=np.float32)

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
                    np.sum(masks * precompute_in[c], axis=agg_axes)
                IN_out = N - np.sum(neg_masks * precompute_out[c], axis=agg_axes)
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

