import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time

import sys  # TODO: remove
sys.path.insert(0, "./src")  # TODO: remove
from contour_depth import Depth, Metric
import contour_depth

if __name__ == "__main__":
	masks = contour_depth.synthetic_data.three_rings(num_masks=250, num_rows=256, num_cols=256, seed=42)
	start = time.time()
	clusters = contour_depth.cluster_inclusion_matrix(masks=masks, num_clusters=3, depth=Depth.EpsilonInclusionDepth, metric=Metric.Depth, kmeans_random_seed=42)
	#clusters = contour_depth.cluster_optimized_eid(masks=masks, num_clusters=3, metric=Metric.Depth, kmeans_random_seed=42)
	print(f"{(time.time() - start)*1000:.2f} ms to compute clusters")

	height, width = masks[0].shape
	out_image = np.zeros((height, width, 3), dtype=np.float32)
	for mask, cluster in zip(masks, clusters):
		grown_mask = scipy.signal.convolve2d(mask, np.ones((3, 3)), mode="same") > 0
		mask_edge = grown_mask - mask
		color = np.random.default_rng(cluster).random(3)
		out_image += np.multiply.outer(mask_edge, color)

	fig, ax = plt.subplots(1, 1)
	ax.imshow(out_image / np.max(out_image))
	plt.show()