from math import comb
import numpy as np
from scipy.optimize import bisect
from enum import Enum

class Depth(Enum):
    EpsilonInclusionDepth = 1
    InclusionDepth = 2
    ContourBandDepth = 3
    ModifiedContourBandDepth = 4


# Computes the contour band depth of an ensemble of contours with J=2.
# Contours are represented as a list of 2 or 3d binary numpy arrays.
# All arrays in the list should have the same dimensions.
# Method is either strict or modified contour band depth.
# If a threshold is specified, the matrix is thresholded as proposed in the CBD paper.
# Otherwise, the use the N^2 method.
# Returns a ndarray of shape (len(masks), ) with the depth score per mask.
# TODO: implement
def contourbanddepth(masks, depth=Depth.ModifiedContourBandDepth, target_mean_mcbd=None):
    num_masks = len(masks)
    num_subsets = comb(num_masks, 2)

    if depth == Depth.ContourBandDepth:  # use inclusion matrix O(N^2)
        depths = []
        inclusion_matrix = compute_inclusion_matrix(masks)
        N_a = np.sum(inclusion_matrix, axis=1)
        N_b = np.sum(inclusion_matrix.T, axis=1)
        N_ab_range = num_masks#N - j_in_cluster
        depths = (N_a * N_b) / (N_ab_range * N_ab_range)
        print(num_subsets)
        print(N_ab_range * N_ab_range)

    elif depth == Depth.ModifiedContourBandDepth:
        depths = []
        bands = compute_band_info(masks)
        for in_mi in masks:
            intersect_subset_ci = []
            ci_subset_union = []

            for band in bands:
                union = band["union"]
                intersection = band["intersection"]

                lc_frac = (intersection - in_mi)
                lc_frac = (lc_frac > 0).sum()
                lc_frac = lc_frac / (intersection.sum() + np.finfo(float).eps)

                rc_frac = (in_mi - union)
                rc_frac = (rc_frac > 0).sum()
                rc_frac = rc_frac / (in_mi.sum() + np.finfo(float).eps)

                intersect_subset_ci.append(lc_frac)
                ci_subset_union.append(rc_frac)

            depths.append((intersect_subset_ci, ci_subset_union))

        depth_matrix_left = np.array([a[0] for a in depths])
        depth_matrix_right = np.array([a[1] for a in depths])

        if target_mean_mcbd is None:  # No threshold  
            depth_matrix_left = 1 - depth_matrix_left
            depth_matrix_right = 1 - depth_matrix_right
            depth_matrix = np.minimum(depth_matrix_left, depth_matrix_right)
        else: # automatically determined threshold as in the paper       
            def mean_depth_deviation(mat, threshold, target):
                return target - (((mat < threshold).astype(float)).sum(axis=1) / num_subsets).mean()
        
            depth_matrix = np.maximum(depth_matrix_left, depth_matrix_right)
            try:
                t = bisect(lambda v: mean_depth_deviation(depth_matrix, v, target_mean_mcbd), depth_matrix.min(),
                        depth_matrix.max())
            except RuntimeError:
                print("Binary search failed to converge")
                t = depth_matrix.mean()

            depth_matrix = (depth_matrix < t).astype(float)

        depths = depth_matrix.mean(axis=1)
    else:
        assert False, f"Unsupported depth {depth}"

    return depths


def compute_band_info(data):
    num_contours = len(data)
    bands = []
    for i in range(num_contours):
        band_a = data[i]
        for j in range(i, num_contours):
            band_b = data[j]
            if i != j:
                subset_sum = band_a + band_b

                band = dict()
                band["union"] = (subset_sum > 0).astype(float)
                band["intersection"] = (subset_sum == 2).astype(float)
                bands.append(band)
    return bands


# Computes the inclusion depth of an ensemble of contours.
# Contours are represented as a list of 2 or 3d binary numpy arrays.
# All arrays in the list should have the same dimensions.
# Method is either strict or epsilon inclusion depth.
# Returns a ndarray of shape (len(masks), ) with the depth score per mask.
# TODO: implement
def inclusiondepth(masks, depth=Depth.EpsilonInclusionDepth):
    num_masks = len(masks)

    if depth == Depth.InclusionDepth:  # use inclusion matrix O(N^2)
        inclusion_matrix = compute_inclusion_matrix(masks)
        N_a = np.sum(inclusion_matrix, axis=1)
        N_b = np.sum(inclusion_matrix.T, axis=1)
        depths = np.minimum(N_a, N_b) / num_masks
    
    elif depth == Depth.EpsilonInclusionDepth:  # use O(N) method
        precompute_in = np.zeros_like(masks[0])
        for in_mi in masks:
            precompute_in += 1 - in_mi
        precompute_out = np.zeros_like(masks[0])
        for in_mi in masks:
            precompute_out += in_mi/in_mi.sum()

        depths = []
        for i in range(num_masks):
            IN_in = num_masks - ((masks[i] / masks[i].sum()) * precompute_in).sum()
            IN_out = num_masks - ((1-masks[i]) * precompute_out).sum()

            # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
            depth = np.minimum((IN_in - 1)/num_masks, (IN_out - 1)/num_masks)

            depths.append(depth)
        
        depths = np.array(depths)        
    else:
        assert False, f"Unsupported depth {depth}"
    
    return depths
    

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
