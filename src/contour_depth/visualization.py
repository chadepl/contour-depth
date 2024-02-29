import numpy as np
import matplotlib.pyplot as plt
from .depth import Depth, compute_band_depth
from .cluster import compute_depth_in_cluster

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
          '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']


def show_spaghetti_plot(masks, iso_value, arr=None, is_arr_categorical=True, vmin=None, vmax=None, cm="magma", ax=None, plot_opts=None):
    """Plots an ensemble of contours using a spaghetti plot.

    Parameters
    ----------
    masks : list
        List of 2d ndarrays, each corresponding to a scalar field.
    iso_value : bool
        Iso value to use to convert the ensemble scalar fields into binary masks.
    arr : ndarray, optional
        Array used for coloring ensemble members, by default None. It can be an array of ints or floats.
    is_arr_categorical : bool, optional
        Used to determine which color map to use, by default True.
    vmin : float, optional
        Lower bound to rescale arr, by default None.
    vmax : float, optional
        Upper bound to rescale arr, by default None.
    cm : str, optional
        Valid matplot lib color map name, by default "magma".
    ax : Axes, optional
        Pyplot Axes, by default None.
    plot_opts : dict, optional
        Dict with plotting options, by default None. Currently supported options are the lines alpha (float [0, 1]) and their linewidth (int > 0).

    Returns
    -------
    Figure
        The figure attached to the ax on which the spaghetti plot was plotted.
    """
    
    num_members = len(masks)

    # Parse plot_opts
    if plot_opts is None:
        plot_opts = dict()
    alpha = plot_opts.get("alpha", 0.5)
    linewidth = plot_opts.get("linewidth", 1)

    # Define coloring
    if arr is not None:
        arr = np.array(arr).flatten()            
    else:
        is_arr_categorical = True
        arr = np.random.choice(np.arange(len(colors)),
                               num_members, replace=True)

    if is_arr_categorical:
        arr = arr.astype(int)
        cs = [colors[e] for e in arr]
    else:         
        if vmin is not None:
            arr = np.clip(arr, a_min=vmin, a_max=arr.max())
        if vmax is not None:
            arr = np.clip(arr, a_min=arr.min(), a_max=vmax)

        if vmin is None and vmax is None:  # scale to fill 0-1 range
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        cs = [plt.get_cmap(cm)(e) for e in arr]

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for i, mask in enumerate(masks):
        ax.contour(mask, levels=[iso_value, ], linewidths=[
                   linewidth, ], colors=[cs[i], ], alpha=alpha)

    return fig


def show_box_plot(masks, depth=Depth.EpsilonInclusionDepth, clustering=None, selected_clusters_ids=None, representative="trimmed_mean", show_out=True, outlier_type="tail", epsilon_out=3, ax=None, plot_opts=None):
    """Plots an ensemble of contours using the contour boxplot idiom.

    Parameters
    ----------
    masks : list
        List of 2d ndarrays, each corresponding to a scalar field.
    depth : Depth, optional
        Type of depth metric to use, by default Depth.EpsilonInclusionDepth
    clustering : ndarray, optional
        Integer ndarray with clustering, by default None. If a clustering is passed, relative depth is used.
    selected_clusters_ids : list, optional
        List of clusters to plot, by default None (then it plots all the clusters). 
    representative : str, optional
        Either median or trimmed_mean, by default "trimmed_mean".
    show_out : bool, optional
        Wheter to show or not outliers, by default True.
    outlier_type : str, optional
        tail, threshold or percent, by default "tail".
    epsilon_out : int, optional
        Depending on the outlier type a integer in [1, N], a depth threshold in [0,1] (float) or a proportion in [0,1] (percent), by default 3.
    ax : Axes, optional
        Pyplot Axes, by default None
    plot_opts : _type_, optional
        Dict with plotting options, by default None. For now no plotting options are available.

    Returns
    -------
    Figure
        The figure attached to the ax on which the spaghetti plot was plotted.
    """
    
    num_contours = len(masks)
    masks_shape = masks[0].shape  # r, c

    if clustering is None:
        clustering = [0 for _ in range(len(masks))]
    clustering = np.array(clustering)

    clusters_ids = np.unique(clustering)
    num_contours_per_cluster = [np.where(clustering == cluster_id)[
        0].size for cluster_id in clusters_ids]

    if clusters_ids.size > 1:
        depths = compute_depth_in_cluster(masks, clustering, clusters_ids.size, depth)
    else:
        depths = compute_band_depth(masks, depth)
    cluster_statistics = __get_bp_depth_elements(
        masks, depths, clustering=clustering, outlier_type=outlier_type, epsilon_out=epsilon_out)

    if selected_clusters_ids is None:
        selected_clusters_ids = clusters_ids.tolist()
    selected_clusters_ids = np.array(selected_clusters_ids)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for cluster_id in selected_clusters_ids:

        if selected_clusters_ids.size == 1:  # traditional boxplot representation
            rep_color = "yellow"
            outliers_color = "red"
            bands_color = {100: "plum", 50: "purple"}
            bands_alpha = {100: 0.3, 50: 0.3}
        else: # multimodal representation
            cluster_color = colors[cluster_id]

            rep_color = cluster_color
            outliers_color = cluster_color
            bands_color = cluster_color
            bands_color = {100: cluster_color, 50: cluster_color}
            bands_alpha = {100: 1 * 0.3, 50: 0.5 * 0.3}

        if show_out and "outliers" in cluster_statistics[cluster_id]:
            cluster_outliers = cluster_statistics[cluster_id]["outliers"]
            for outlier_id, outlier_mask in zip(cluster_outliers["idx"], cluster_outliers["masks"]):
                ax.contour(outlier_mask, levels=[0.5,], colors=[outliers_color, ], linewidths=[
                           1,], linestyles=["dashed",], alpha=0.8)

        if "bands" in cluster_statistics[cluster_id]:
            cluster_bands = cluster_statistics[cluster_id]["bands"]

            for i, (bid, bmask, bweight) in enumerate(zip(cluster_bands["idx"], cluster_bands["masks"], cluster_bands["weights"])):
                ax.contourf(bmask, levels=[0.5, 1.5], colors=[
                            bands_color[bweight], ], alpha=bands_alpha[bweight])

        if "representatives" in cluster_statistics[cluster_id]:
            if representative == "median":
                rep_mask = cluster_statistics[cluster_id]["representatives"]["masks"][0]
            elif representative == "trimmed_mean":
                rep_mask = cluster_statistics[cluster_id]["representatives"]["masks"][1]
            else:
                assert False, f"representative can be either median or trimmed_mean"
            ax.contour(rep_mask, levels=[0.5,], colors=[
                       rep_color, ], linewidths=[3,])

    # Add legend clusters + proportions
    if selected_clusters_ids.size > 1:
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
            bar_height = RECT_HEIGHT * \
                (num_contours_per_cluster[cluster_id]/num_contours)
            rect = Rectangle((BAR_X0, bar_y0), RECT_WIDTH,
                            bar_height, color=cluster_color, edgecolor=None)
            ax.add_patch(rect)
            bar_y0 += bar_height

    return fig


def __get_bp_depth_elements(masks, depths, clustering=None, outlier_type="tail", epsilon_out=3) -> dict:
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
        cluster_statistics[cluster_id]["representatives"] = dict(
            idx=[median_coord, ], masks=[median_mask, ])

        # outliers
        if outlier_type == "threshold":
            outliers_idx = np.where(subset_depths <= epsilon_out)[
                0]  # should be 0
        elif outlier_type == "tail":
            outliers_idx = np.argsort(subset_depths)[
                :int(epsilon_out)]  # should be 0
        elif outlier_type == "percent":
            outliers_idx = np.argsort(subset_depths)[
                :int(subset_depths.size*epsilon_out)]
        outliers_coords = [coords[oid] for oid in outliers_idx]
        cluster_statistics[cluster_id]["outliers"] = dict(idx=[], masks=[])
        for ocoord in outliers_coords:
            cluster_statistics[cluster_id]["outliers"]["idx"].append(ocoord)
            cluster_statistics[cluster_id]["outliers"]["masks"].append(
                masks[ocoord])

        # bands

        sorted_depths_idx = np.argsort(subset_depths)[::-1]
        band100_idx = sorted_depths_idx[~np.in1d(
            sorted_depths_idx, outliers_idx)]
        band50_idx = band100_idx[:band100_idx.size // 2]

        band100_coords = [coords[bid] for bid in band100_idx]
        band50_coords = [coords[bid] for bid in band50_idx]

        if len(band100_coords) >= 2:
            band100_mask = np.array([masks[bcoord]
                                    for bcoord in band100_coords]).sum(axis=0)
            new_band100_mask = np.zeros_like(band100_mask)
            new_band100_mask[band100_mask == 0] = 2  # outside
            new_band100_mask[band100_mask > 0] = 1  # in the band
            new_band100_mask[band100_mask == len(band100_coords)] = 0  # inside
            band100_mask = new_band100_mask
        else:
            band100_mask = np.zeros_like(masks[0])  # TODO: should be None?

        if len(band50_coords) >= 2:
            band50_mask = np.array([masks[bcoord]
                                   for bcoord in band50_coords]).sum(axis=0)
            new_band50_mask = np.zeros_like(band50_mask)
            new_band50_mask[band50_mask == 0] = 2  # outside
            new_band50_mask[band50_mask > 0] = 1  # in the band
            new_band50_mask[band50_mask == len(band50_coords)] = 0  # inside
            band50_mask = new_band50_mask
        else:
            band50_mask = np.zeros_like(masks[0])   # TODO: should be None?

        cluster_statistics[cluster_id]["bands"] = dict(
            idx=["b100", "b50"], masks=[band100_mask, band50_mask], weights=[100, 50])

        # trimmed mean
        mean_coord = -1  # mean does not have a coordinate
        mean_mask = (np.array([masks[bcoord] for bcoord in band100_coords]).mean(axis=0) > 0.5).astype(float)
        cluster_statistics[cluster_id]["representatives"]["idx"].append(mean_coord)
        cluster_statistics[cluster_id]["representatives"]["masks"].append(mean_mask)

    return cluster_statistics
