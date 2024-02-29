import math
import numpy as np
from enum import Enum


class Depth(Enum):
    ContourBandDepth = 1
    InclusionDepth = 2
    EpsilonContourBandDepth = 3
    EpsilonInclusionDepth = 4


def compute_band_depth(masks, depth: Depth):
    if depth == Depth.ContourBandDepth:
        return compute_exact_contour_band_depth(masks)
    elif depth == Depth.InclusionDepth:
        return compute_inclusion_depth(masks)
    elif depth == Depth.EpsilonContourBandDepth:
        return compute_epsilon_contour_band_depth(masks)
    elif depth == Depth.EpsilonInclusionDepth:
        return compute_epsilon_inclusion_depth(masks)
    else:
        assert False, f"Unknown depth type {depth}"


def compute_exact_contour_band_depth(masks):
    num_subsets = math.comb(len(masks) - 1, 2)
    masks = np.array(masks, dtype=np.uint32)
    inclusion_matrix = compute_inclusion_matrix(masks)
    N_a = np.sum(inclusion_matrix, axis=1)
    N_b = np.sum(inclusion_matrix.T, axis=1)
    return (N_a * N_b) / num_subsets


def compute_epsilon_contour_band_depth(masks, epsilon=None):
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


def compute_inclusion_depth(masks):
    num_masks = len(masks)
    inclusion_matrix = compute_inclusion_matrix(masks)
    N_a = np.sum(inclusion_matrix, axis=1)
    N_b = np.sum(inclusion_matrix.T, axis=1)
    return np.minimum(N_a, N_b) / num_masks


def compute_epsilon_inclusion_depth(masks):
    masks = np.array(masks)
    inverted_masks = 1 - masks
    area_normalized_masks = (masks.T / np.sum(masks, axis=(1, 2)).T).T
    precompute_in = np.sum(inverted_masks, axis=0)
    precompute_out = np.sum(area_normalized_masks, axis=0)

    num_masks = len(masks)
    IN_in = num_masks - np.sum(area_normalized_masks * precompute_in, axis=(1, 2))
    IN_out = num_masks - np.sum(inverted_masks * precompute_out, axis=(1, 2))
    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    return (np.minimum(IN_in, IN_out) - 1) / len(masks)


def compute_inclusion_matrix(masks):
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all(
            (masks & masks[i]) == masks[i], axis=(1, 2))
        inclusion_mat[i, i] = 0
    return inclusion_mat


def compute_epsilon_inclusion_matrix(masks):
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    for i in range(num_masks):
        inclusion_mat[i, :] = 1 - \
            np.sum(inv_masks & masks[i], axis=(1, 2)) / np.sum(masks[i])
        inclusion_mat[i, i] = 0
    return inclusion_mat
