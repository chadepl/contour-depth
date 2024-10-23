import math
import numpy as np
from enum import Enum


def _get_agg_axes(masks_arr:np.array):
    """Returns the axes along which methods should integrate.
    The fist axis is assummed to be the ensemble axis.

    Parameters
    ----------
    masks_arr : np.array
        Contour ensemble as a numpy array. First axis corresponds to the ensemble dimension.

    Returns
    -------
    tuple
        Axes along which to integrate.
    """
    n_dim = len(masks_arr.shape)
    if n_dim > 1: # Needs at least the ensemble dimension + another one
        return tuple(list(range(n_dim))[1:])
    else:
        assert False, "Too few dimensions to perform contour depth analysis."


class Depth(Enum):
    ContourBandDepth = 1
    InclusionDepth = 2
    EpsilonContourBandDepth = 3
    EpsilonInclusionDepth = 4


def compute_contour_depth(masks:list[np.array], depth:Depth, kwargs:dict=None) -> np.array:
    """Computes the contour depth of a contour ensemble.

    Parameters
    ----------
    masks : list[np.array]
        List of binary masks representing the contours.
    depth : Depth
        Choice of depth notion (see the Depth enum for available options). 
    kwargs : dict
        Keyword parameters that the selected depth notion supports.

    Returns
    -------
    np.array
        Array containing the depth of each contour.
    """
    if not kwargs:
        kwargs = dict()
    if depth == Depth.ContourBandDepth:
        return compute_exact_contour_band_depth(masks, **kwargs)
    elif depth == Depth.InclusionDepth:
        return compute_inclusion_depth(masks, **kwargs)
    elif depth == Depth.EpsilonContourBandDepth:
        return compute_epsilon_contour_band_depth(masks, **kwargs)
    elif depth == Depth.EpsilonInclusionDepth:
        return compute_epsilon_inclusion_depth(masks, **kwargs)
    else:
        assert False, f"Unknown depth type {depth}"


def compute_exact_contour_band_depth(masks:list[np.array]) -> np.array:
    num_subsets = math.comb(len(masks) - 1, 2)
    masks = np.array(masks, dtype=np.uint32)    
    inclusion_matrix = compute_inclusion_matrix(masks)
    N_a = np.sum(inclusion_matrix, axis=1)
    N_b = np.sum(inclusion_matrix.T, axis=1)
    return (N_a * N_b) / num_subsets


def compute_epsilon_contour_band_depth(masks:list[np.array], epsilon:float=None) -> np.array:
    # https://users.cs.utah.edu/~kirby/Publications/Kirby-82.pdf
    def epsilon_subset_operator(A, B):
        area_A = np.sum(A)
        if area_A == 0:
            return 0
        else:
            area_A_min_B = np.sum(A & (~B))
            return area_A_min_B / area_A

    masks = np.array(masks, dtype=np.uint32)
    num_masks = len(masks)
    num_subsets = math.comb(num_masks - 1, 2)
    epsilon_matrix = np.zeros((num_masks, num_subsets))
    for i, mask in enumerate(masks):
        subset_idx = 0
        for j1, j1_mask in enumerate(masks):
            for j2, j2_mask in enumerate(masks[j1+1:]):
                if j1 == i or j1+j2+1 == i:
                    continue

                mask_intersection = j1_mask & j2_mask
                mask_union = j1_mask | j2_mask
                res_intersection = epsilon_subset_operator(mask_intersection, mask)
                res_union = epsilon_subset_operator(mask, mask_union)
                epsilon_matrix[i, subset_idx] = max(res_intersection, res_union)
                subset_idx += 1

    if not epsilon:
        min_per_column = np.min(epsilon_matrix, axis=1)
        # min_per_column = epsilon_matrix.flatten()
        epsilon = sorted(min_per_column)[num_masks // 6]

    return np.sum(epsilon_matrix <= epsilon, axis=1) / num_subsets


def compute_inclusion_depth(masks:list[np.array]) -> np.array:
    num_masks = len(masks)
    inclusion_matrix = compute_inclusion_matrix(masks)
    N_a = np.sum(inclusion_matrix, axis=1)
    N_b = np.sum(inclusion_matrix.T, axis=1)
    return np.minimum(N_a, N_b) / num_masks


def compute_epsilon_inclusion_depth(masks:list[np.array]) -> np.array:
    masks = np.array(masks)
    agg_axes = _get_agg_axes(masks)

    inverted_masks = 1 - masks
    area_normalized_masks = (masks.T / np.sum(masks, axis=agg_axes).T).T
    precompute_in = np.sum(inverted_masks, axis=0)
    precompute_out = np.sum(area_normalized_masks, axis=0)

    num_masks = len(masks)
    IN_in = num_masks - np.sum(area_normalized_masks * precompute_in, axis=agg_axes)
    IN_out = num_masks - np.sum(inverted_masks * precompute_out, axis=agg_axes)
    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    return (np.minimum(IN_in, IN_out) - 1) / len(masks)


def compute_inclusion_matrix(masks:list[np.array]) -> np.array:
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    agg_axes = _get_agg_axes(masks)
    inclusion_mat = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all(
            (masks & masks[i]) == masks[i], axis=agg_axes)
        inclusion_mat[i, i] = 0
    return inclusion_mat


def compute_epsilon_inclusion_matrix(masks:list[np.array]) -> np.array:
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    agg_axes = _get_agg_axes(masks)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    for i in range(num_masks):
        inclusion_mat[i, :] = 1 - \
            np.sum(inv_masks & masks[i], axis=agg_axes) / np.sum(masks[i])
        inclusion_mat[i, i] = 0
    return inclusion_mat
