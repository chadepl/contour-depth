from time import time
from pathlib import Path
from functools import reduce
import napari.layers
import nibabel as nib

import numpy as np

from skimage.morphology import dilation
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.filters import gaussian

from data_loader import load_case
from contour_depth import Depth
from contour_depth import cluster_optimized_eid, compute_depth_in_cluster, compute_relative_depth

import napari
import distinctipy


#######################
# Settable parameters #
#######################

ANA_REGION_ID = 4
ROI_ID = 1

ANA_REGION = ["GI", "GYN", "Breast", "Sarcoma", "H&N"][ANA_REGION_ID]
ROI = {
    'GI': ['Bag_Bowel', 'CTV_4500', 'CTV_5400'], 
    'GYN': ['CTVn_4500', 'GTVn', 'Bowel_Small', 'CTVp_4500'], 
    'Breast': ['CTV_Sclav_LN', 'Heart', 'BrachialPlex_L', 'CTV_Ax', 'CTV_IMN', 'A_LAD', 'CTV_Chestwall'], 
    'Sarcoma': ['Genitals', 'CTV', 'GTV'], 
    'H&N': ['Brainstem', 'Parotid_L', 'Parotid_R', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Larynx', 'Musc_Constrict', 'GTVp', 'GTVn', 'CTV1', 'CTV2']}[ANA_REGION][ROI_ID]

DEPTH = Depth.EpsilonInclusionDepth
NUM_CLUSTERS = 1
NUM_OUTLIERS = 5

CBP_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"] # six available colors for when we have clusterings    
UNI_MODAL_CBP_COLORS = dict(median="yellow", mean="teal", hundred_band=(103/255,49/255,71/255), fifty_band=(160/255,32/255,240/255))

ENSEMBLE_TYPE = "expert" # [expert, non expert]
CONSENSUS = "expert" # [expert, non expert]

PLOT_SPAGHETTI = False # Caution: slow

#############
# Load data #
#############
print("Loading data ...")

img, yexpert_consensus, yexpert_segs = load_case(ana_region=ANA_REGION, roi=ROI, use_expert_segs=True)
_, nexpert_consensus, nexpert_segs = load_case(ana_region=ANA_REGION, roi=ROI, use_expert_segs=False)

segs = nexpert_segs
print(" - Num segs", len(segs))

spatial_domain = img.shape

# we extract the subset of the spatial domain where the ROI is to spare computation
def get_segs_spatial_subset(segs):
    seg_envelope = reduce(lambda a, b: a + b, segs)
    seg_envelope = (seg_envelope >= 1).astype(int)
    seg_envelope = dilation(seg_envelope)
    props = regionprops(seg_envelope)
    if len(props) > 1:
        raise Exception(f"Num regions > 1 not implemented (Num regions is {len(props)})")
    bbox = props[0]["bbox"] # min_a0, min_a1, min_a2, max_a0, max_a1, max_a2
    segs_subset = [seg[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] for seg in segs]
    return segs_subset, bbox

segs_subset, bbox = get_segs_spatial_subset(segs)


###################
# Contour boxplot #
###################
print("Computing Contour Box Plot data ...")

t_start = time()
if NUM_CLUSTERS > 1:
    segs_labels = cluster_optimized_eid(segs_subset, num_clusters=NUM_CLUSTERS)
else:
    segs_labels = np.zeros(len(segs_subset), dtype=int)
t_end = time()
print(f" - Finished clustering (num_cluster={NUM_CLUSTERS}) in {(t_end - t_start)} seconds")

t_start = time()
segs_cluster_depths = compute_depth_in_cluster(segs_subset, segs_labels, num_clusters=NUM_CLUSTERS, depth=DEPTH)
if NUM_CLUSTERS > 1:
    red_all = compute_relative_depth(segs_cluster_depths, num_clusters=NUM_CLUSTERS)
    red = np.zeros(len(segs_subset))
    red[np.where(segs_labels== 0)] = red_all[0][np.where(segs_labels== 0)]
    red[np.where(segs_labels== 1)] = red_all[0][np.where(segs_labels== 1)]
else:
    red = segs_cluster_depths.copy().flatten()
t_end = time()
print(f" - Finished computing relative depths in {(t_end - t_start)} seconds")

# Obtains ensemble elements like median, mean and confidence bands with the depth information
def get_cbps_components(depths, masks, num_outliers = 0, clustering=None, spatial_domain=None, bbox=None):

    if clustering is None:
        clustering = np.zeros(len(depths), dtype=int)

    segs_arr = np.array(masks)
    cpb_components = dict()
    
    for clustering_lab in np.unique(clustering):

        print(f"Processing cluster {clustering_lab}")

        cluster_idx = np.where(clustering == clustering_lab)[0]
        cluster_depths = depths[cluster_idx]
        cluster_segs = segs_arr[cluster_idx]
        depths_argsort = np.argsort(cluster_depths)[::-1]

        if num_outliers == 0:
            outliers_idx = np.array([])
            hundred_segs_idx = depths_argsort
        else:
            outliers_idx = depths_argsort[-num_outliers:]
            hundred_segs_idx = depths_argsort[:-num_outliers]

        fifty_segs_idx = depths_argsort[:hundred_segs_idx.size//2]

        print(f" -- Cluster {clustering_lab}")
        print("  --- Total num segs: ", len(cluster_segs))
        print("  --- Num segs 100% band: ", len(hundred_segs_idx))
        print("  --- Num segs 50% band: ", len(fifty_segs_idx))
        print("  --- Num outliers: ", outliers_idx.size)

        # hundred_band = cluster_segs[hundred_segs_idx].sum(axis=0)
        hundred_band = reduce(lambda a, b: a + b, [cluster_segs[seg_i] for seg_i in hundred_segs_idx])
        hundred_band[np.logical_or(hundred_band == 0, hundred_band == len(hundred_segs_idx))] = 0
        hundred_band[hundred_band>0] = 1

        # fifty_band = cluster_segs[fifty_segs_idx].sum(axis=0)
        fifty_band = reduce(lambda a, b: a + b, [cluster_segs[seg_i] for seg_i in fifty_segs_idx])
        fifty_band[np.logical_or(fifty_band == 0, fifty_band == len(fifty_segs_idx))] = 0
        fifty_band[fifty_band>0] = 1

        # mean = cluster_segs[hundred_segs_idx].mean(axis=0)
        mean = reduce(lambda a, b: a + b, [cluster_segs[seg_i] for seg_i in hundred_segs_idx]) / len(hundred_segs_idx)
        mean_mask = (mean >= 0.5).astype(int)

        median = depths_argsort[0]#np.argsort(cluster_depths)[::-1][0]
        median_mask = cluster_segs[median]

        if spatial_domain and bbox:
            new_fifty_band = np.zeros(spatial_domain, dtype=int)
            new_fifty_band[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = fifty_band
            fifty_band = new_fifty_band
            new_hundred_band = np.zeros(spatial_domain, dtype=int)
            new_hundred_band[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = hundred_band
            hundred_band = new_hundred_band
            new_median_mask = np.zeros(spatial_domain, dtype=int)
            new_median_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = median_mask
            median_mask = new_median_mask
            new_mean_mask = np.zeros(spatial_domain, dtype=int)
            new_mean_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = mean_mask
            mean_mask = new_mean_mask

        cpb_components[clustering_lab] = dict(
            median = median_mask,
            mean = mean_mask,
            hundred_band = hundred_band,
            fifty_band = fifty_band,
        )

    return cpb_components

t_start = time()
cbps_components = get_cbps_components(red, segs_subset, num_outliers=NUM_OUTLIERS, clustering=segs_labels, spatial_domain=spatial_domain, bbox=bbox)
t_end = time()
print(f" - Finished computing cbp components in {(t_end - t_start)} seconds")


#################
# Visualization #
#################
print("Initializing GUI ...")

PLOT_UNIMODAL_CBP = not len(cbps_components) > 1
print("PLOT_UNIMODAL_CBP", PLOT_UNIMODAL_CBP)

yexpert_consensus_color_map = napari.utils.DirectLabelColormap(color_dict={1:(1,1,1)})
nexpert_consensus_color_map = napari.utils.DirectLabelColormap(color_dict={1:(1,0,0)})
hundred_color_map = napari.utils.DirectLabelColormap(color_dict={1:(103/255,49/255,71/255)})
fifty_color_map = napari.utils.DirectLabelColormap(color_dict={1:(160/255,32/255,240/255)})
median_color_map = napari.utils.DirectLabelColormap(color_dict={1:(1,1,0)})
mean_color_map = napari.utils.DirectLabelColormap(color_dict={1:(0,0,1)})

viewer = napari.Viewer()
viewer.title = f"{ANA_REGION} - {ROI}"

viewer.add_image(img)

# contour boxplot
if PLOT_UNIMODAL_CBP:
    median_mask, mean_mask, fifty_band, hundred_band = cbps_components[0]["median"], cbps_components[0]["mean"], cbps_components[0]["fifty_band"], cbps_components[0]["hundred_band"]
    viewer.add_labels(hundred_band, colormap=hundred_color_map, name="hundred band")
    viewer.add_labels(fifty_band, colormap=fifty_color_map, name="fifty band")
    labs = viewer.add_labels(median_mask, colormap=median_color_map, name="median")
    labs.contour = 1
    labs = viewer.add_labels(mean_mask, colormap=mean_color_map, name="mean")
    labs.contour = 1
else:
    for cluster_lab, cbp_components in cbps_components.items():
        cbp_color = CBP_COLORS[cluster_lab]
        cbp_color_map = napari.utils.DirectLabelColormap(color_dict={1:cbp_color})
        median_mask, mean_mask, fifty_band, hundred_band = cbp_components["median"], cbp_components["mean"], cbp_components["fifty_band"], cbp_components["hundred_band"]
        viewer.add_labels(hundred_band, colormap=cbp_color_map, opacity=0.2, name="hundred band")
        viewer.add_labels(fifty_band, colormap=cbp_color_map, opacity=0.6, name="fifty band")
        labs = viewer.add_labels(mean_mask, colormap=cbp_color_map, name="mean")
        labs.contour = 1
        # labs = viewer.add_labels(mean_mask, colormap=cbp_color_map, name="mean")
        # labs.contour = 1

layer = napari.layers.Labels(yexpert_consensus, colormap=yexpert_consensus_color_map, name="expert consensus")
layer.contour = 1
viewer.add_layer(layer)
layer = napari.layers.Labels(nexpert_consensus, colormap=nexpert_consensus_color_map, name="non-expert consensus")
layer.contour = 1
viewer.add_layer(layer)

# Plot spaghetti plot
if PLOT_SPAGHETTI:
    def plot_spaghetti(slice_num):        
        for cluster_id in np.unique(segs_labels):
            idx = np.where(segs_labels == cluster_id)[0]
            if PLOT_UNIMODAL_CBP:
                shape_colors = []
                colors = distinctipy.get_colors(idx.size)
            else:
                shape_colors = CBP_COLORS[cluster_id]
            shapes = []
            
            for si in idx:
                min_a0, min_a1, min_a2, max_a0, max_a1, max_a2 = bbox # to offset
                mask = segs_subset[si][slice_num - min_a0].astype(float)
                bmask = gaussian(gaussian(mask, sigma=1), sigma=1)
                contours = find_contours(bmask, level=0.5)
                if len(contours) > 0:
                    shapes.append(contours[0] + np.array([min_a1, min_a2]))
                    if PLOT_UNIMODAL_CBP:
                        shape_colors.append(colors[si])                    
            viewer.add_shapes(data=shapes, shape_type="polygon", face_color=[0,0,0,0], edge_color=shape_colors, edge_width=0.5, name=f"spaghetti clust={cluster_id}")

    prev_slice = None
    curr_slice = int(viewer.dims.point[0])

    plot_spaghetti(curr_slice)

    @viewer.dims.events.current_step.connect
    def slice_change():
        global prev_slice, curr_slice

        spaghetti_layers = [l for l in viewer.layers if "spaghetti" in l.name]
        if len(spaghetti_layers) > 0:
            for l in spaghetti_layers:
                viewer.layers.remove(l)

        prev_slice = curr_slice
        curr_slice = int(viewer.dims.point[0])    
        t_start = time()
        plot_spaghetti(curr_slice)
        t_end = time()
        print(f" - Took {(t_end - t_start)} seconds to update spaghetti")

napari.run()