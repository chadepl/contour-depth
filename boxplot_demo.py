# we compute the depths using the different methods
# then we plot them using a boxplot
import matplotlib.pyplot as plt
import numpy as np

import contour_depth
from contour_depth import Depth

if __name__ == "__main__":
    masks, labels = contour_depth.synthetic_data.three_rings(num_masks=250, num_rows=256, num_cols=256, seed=42, return_labels=True)
    # masks = contour_depth.synthetic_data.circle_ensemble(num_masks=25, num_rows=256, num_cols=256, radius_std=0.1, seed=42); labels = None

    fig, axs = plt.subplots(ncols=3)

    axs[0].set_title("Spaghetti plot")
    contour_depth.visualization.show_spaghetti_plot(masks, 0.5, ax=axs[0])

    axs[1].set_title("CBD")
    contour_depth.visualization.show_box_plot(masks, clustering=labels, depth=Depth.ContourBandDepth, ax=axs[1])

    axs[2].set_title("ID")
    contour_depth.visualization.show_box_plot(masks, clustering=labels, depth=Depth.InclusionDepth, ax=axs[2])

    plt.show()