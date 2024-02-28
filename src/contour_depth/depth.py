import numpy as np
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
def contourbanddepth(masks, method=""):
    pass


# Computes the inclusion depth of an ensemble of contours.
# Contours are represented as a list of 2 or 3d binary numpy arrays.
# All arrays in the list should have the same dimensions.
# Method is either strict or epsilon inclusion depth.
# Returns a ndarray of shape (len(masks), ) with the depth score per mask.
# TODO: implement
def inclusiondepth(masks, method=""):
    N_a = np.sum(inclusion_matrix[:, j_in_cluster], axis=1)
    N_b = np.sum(inclusion_matrix.T[:, j_in_cluster], axis=1)
    depth_in_cluster[c] = np.minimum(N_a, N_b) / N
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
