# we compute the depths using the different methods
# then we plot them using a boxplot
import matplotlib.pyplot as plt

import sys  # TODO: remove
sys.path.insert(0, "./src")  # TODO: remove
import contour_depth
from contour_depth import Depth


if __name__ == "__main__":
    masks = contour_depth.synthetic_data.circle_ensemble(num_masks=250, num_rows=256, num_cols=256, seed=42)

    fig, axs = plt.subplots(ncols=3)

    axs[0].set_title("Spaghetti plot")
    contour_depth.visualization.spaghettiplot(masks, 0.5, ax=axs[0])

    axs[1].set_title("mCBD")
    contour_depth.visualization.contourboxplot(masks, depth=Depth.ContourBandDepth, ax=axs[1])

    axs[2].set_title("eID")
    contour_depth.visualization.contourboxplot(masks, depth=Depth.InclusionDepth, ax=axs[2])

    plt.show()