import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .depth import Depth, contourbanddepth, inclusiondepth

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']


# Plots each contour in a different color.
# Returns newly created figure, otherwise the figure to which the ax is connected.
def spaghettiplot(masks, iso_value, arr=None, is_arr_categorical=True, vmin=None, vmax=None, ax=None, plot_opts=None):
    num_members = len(masks)

    if plot_opts is None:
        plot_opts = dict()    
    alpha = plot_opts.get("alpha", 0.5)
    linewidth = plot_opts.get("linewidth", 1)

    if arr is not None:
        arr = np.array(arr).flatten()
        if is_arr_categorical:
            arr = arr.astype(int)
    else:
        is_arr_categorical = True
        arr = np.random.choice(np.arange(len(colors)), num_members, replace=True)

    if is_arr_categorical:
        cs = [colors[e] for e in arr]
    else:
        arr = np.array(arr)
        if vmin is not None:
            arr = np.clip(arr, a_min=vmin, a_max=arr.max())
        if vmax is not None:
            arr = np.clip(arr, a_min=arr.min(), a_max=vmax)

        if vmin is None and vmax is None:  # scale to fill 0-1 range
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        cs = [cm.magma(e) for e in arr]

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(layout="tight", figsize=(10, 10))

    for i, mask in enumerate(masks):
        ax.contour(mask, levels=[iso_value, ], linewidths=[linewidth, ], colors=[cs[i], ], alpha=alpha)

    ax.set_axis_off()

    if ax_was_none:
        plt.show()
    else:
        return ax


# masks, labs, method="depth", method_kwargs=dict(),
                        #  cluster_statistics = None,
                        #  focus_clusters=None,
                        #  show_out=True, under_mask=None,
                        #  smooth=True, smooth_its=1, smooth_kernel_size=1, axis_off=True,
                        #  ax=None

# Plots contour boxplot based on the method (depth notion).
# The boxplot supports multi-modal contour boxplots.
# If a labeling is specified, the depths are the relative depths with respect to each cluster as in the reference [CITE].
# User specificies the outlier method, either depth threshold, percentage or tail.
# For now, this method does not support overlaying an image.
# Returns newly created figure, otherwise the figure to which the ax is connected.
def contourboxplot(masks, depth=Depth.EpsilonInclusionDepth, clustering=None, selected_clusters_ids=None, show_out=True, outlier_type="tail", epsilon_out=3, ax=None, plot_opts=None):
    num_contours = len(masks)    

    # TODO: verify all masks have the same shape
    masks_shape = masks[0].shape  # r, c

    if clustering is None:
        clustering = [0 for _ in range(len(masks))]
    clustering = np.array(clustering)

    clusters_ids = np.unique(clustering)
    num_contours_per_cluster = [np.where(clustering == cluster_id)[0].size for cluster_id in clusters_ids]

    if depth == Depth.ContourBandDepth or depth == Depth.ModifiedContourBandDepth:
        depths = np.zeros(len(masks))  # TODO
    elif depth == Depth.InclusionDepth or depth == Depth.EpsilonInclusionDepth:
        depths = np.zeros(len(masks))  # TODO
    else:
        assert False, f"Unsupported depth {depth}"
    cluster_statistics = get_bp_depth_elements(masks, depths, clustering=clustering, outlier_type=outlier_type, epsilon_out=epsilon_out)

    if selected_clusters_ids is None:
        selected_clusters_ids = clusters_ids.tolist()
    selected_clusters_ids = np.array(selected_clusters_ids)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")

    for cluster_id in selected_clusters_ids:
        cluster_color = colors[cluster_id] # plum, purple, yellow, red and teal where used before        

        median_color = cluster_color
        outliers_color = cluster_color
        bands_color = cluster_color        

        if show_out and "outliers" in cluster_statistics[cluster_id]:
            cluster_outliers = cluster_statistics[cluster_id]["outliers"]
            for outlier_id, outlier_mask in zip(cluster_outliers["idx"], cluster_outliers["masks"]):
                ax.contour(outlier_mask, levels=[0.5,], colors=[outliers_color, ], linewidths=[1,], linestyles=["dashed",], alpha=0.8)
        
        if "bands" in cluster_statistics[cluster_id]:
            cluster_bands = cluster_statistics[cluster_id]["bands"]            

            for i, (bid, bmask, bweight) in enumerate(zip(cluster_bands["idx"], cluster_bands["masks"], cluster_bands["weights"])):                                
                ax.contourf(bmask, levels=[0.5, 1.5], colors=[bands_color, ], alpha=(bweight/100) * 0.3)

        if "representatives" in cluster_statistics[cluster_id]:
            median_mask = cluster_statistics[cluster_id]["representatives"]["masks"][0]
            ax.contour(median_mask, levels=[0.5,], colors=[median_color, ], linewidths=[3,])

    # Add legend bar    
    print("shape", masks_shape)
    from matplotlib.patches import Rectangle
    OFFSET_R = 0.02 * masks_shape[1]  # distance from right side
    PADDING_TB = 0.04 * masks_shape[0]  # padding top bottom    
    RECT_HEIGHT = masks_shape[0] - PADDING_TB
    RECT_WIDTH = 0.05 * masks_shape[1]
    BAR_X0 = masks_shape[1] - RECT_WIDTH - OFFSET_R
    bar_y0 = PADDING_TB/2
    for cluster_id in clusters_ids:
        cluster_color = colors[cluster_id]
        if cluster_id not in selected_clusters_ids:
            cluster_color = "lightgray"
        bar_height = RECT_HEIGHT*(num_contours_per_cluster[cluster_id]/num_contours)        
        rect = Rectangle((BAR_X0, bar_y0), RECT_WIDTH, bar_height, color=cluster_color, edgecolor=None)
        ax.add_patch(rect)
        bar_y0 += bar_height

    return ax


def get_bp_depth_elements(masks, depths, clustering=None, outlier_type="tail", epsilon_out=3) -> dict:
    # returns per cluster: representatives, bands and outliers

    depths = np.array(depths).flatten()
    if clustering is None:
        clustering = [0 for _ in range(depths.size)]
    clustering = np.array(clustering)
    clusters_ids = np.unique(clustering)

    cluster_statistics = dict()    

    for cluster_id in clusters_ids:
        cluster_statistics[cluster_id] = dict()

        coords = np.where(clustering == cluster_id)[0]
        subset_depths = depths[coords]

        # representatives        
        median_id = np.argmax(subset_depths)
        median_coord = coords[median_id]
        median_mask = masks[median_coord]
        cluster_statistics[cluster_id]["representatives"] = dict(idx=[median_coord, ], masks=[median_mask, ])
        
        # outliers
        if outlier_type == "threshold":
            outliers_idx = np.where(subset_depths <= epsilon_out)[0]  # should be 0
        elif outlier_type == "tail":
            outliers_idx = np.argsort(subset_depths)[:int(epsilon_out)]  # should be 0
        elif outlier_type == "percent":
            outliers_idx = np.argsort(subset_depths)[:int(subset_depths.size*epsilon_out)]
        outliers_coords = [coords[oid] for oid in outliers_idx]
        cluster_statistics[cluster_id]["outliers"] = dict(idx=[], masks=[])
        for ocoord in outliers_coords:
            cluster_statistics[cluster_id]["outliers"]["idx"].append(ocoord)
            cluster_statistics[cluster_id]["outliers"]["masks"].append(masks[ocoord])

        # bands

        sorted_depths_idx = np.argsort(subset_depths)[::-1]
        band100_idx = sorted_depths_idx[~np.in1d(sorted_depths_idx, outliers_idx)]
        band50_idx = band100_idx[:band100_idx.size // 4]        

        band100_coords = [coords[bid] for bid in band100_idx]
        band50_coords = [coords[bid] for bid in band50_idx]

        if len(band100_coords) >= 2:
            band100_mask = np.array([masks[bcoord] for bcoord in band100_coords]).sum(axis=0)
            new_band100_mask = np.zeros_like(band100_mask)
            new_band100_mask[band100_mask == 0] = 2  # outside
            new_band100_mask[band100_mask > 0] = 1  # in the band
            new_band100_mask[band100_mask == len(band100_coords)] = 0  # inside
            band100_mask = new_band100_mask
        else:
            band100_mask = np.zeros_like(masks[0])  # TODO: should be None?

        if len(band50_coords) >= 2:
            band50_mask = np.array([masks[bcoord] for bcoord in band50_coords]).sum(axis=0)
            new_band50_mask = np.zeros_like(band50_mask)
            new_band50_mask[band50_mask == 0] = 2  # outside
            new_band50_mask[band50_mask > 0] = 1  # in the band
            new_band50_mask[band50_mask == len(band50_coords)] = 0  # inside
            band50_mask = new_band50_mask
        else:
            band50_mask = np.zeros_like(masks[0])   # TODO: should be None?

        cluster_statistics[cluster_id]["bands"] = dict(idx=["b100", "b50"], masks=[band100_mask, band50_mask], weights=[100, 50])
        
        # trimmed mean
        # masks_arr = np.array([m.flatten() for m in [masks[i] for i in cbp_band100]])
        # masks_mean = masks_arr.mean(axis=0)
        # contours = find_contours(masks_mean.reshape(masks[0].shape), level=0.5)
        # plot_contour(contours, line_kwargs=dict(c="dodgerblue", linewidth=5), smooth_line=smooth_line, ax=ax)

    return cluster_statistics