import numpy as np
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from scipy import ndimage
import networkx as nx
from collections import defaultdict
from time import time
import cv2
from skimage import io


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]


def get_quantiles(x):
    quants = np.quantile(
        x, [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # quants = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    # print(x.shape)
    # exit()
    # quants = np.quantile(x, [0, 0.5, 1])
    return quants


# Combines the functionality of determining labels and features for each supervoxel with discarding supervoxels that lie outside of the brain.
def determine_nodes_and_features(supervoxel_partitioning, voxel_intensities, voxel_labels, num_supervoxels):
    new_region_img, node_feats, node_centroids, node_labels = extract_supervoxel_statistics(
        supervoxel_partitioning, voxel_intensities, voxel_labels, num_supervoxels)
    return new_region_img, node_feats, node_centroids, node_labels


def extract_supervoxel_statistics(sv_partitioning, voxel_intensities, voxel_labels, num_supervoxels):
    # first check how many incoming channels there are

    sv_feats = get_sv_summary_for_modality(
        voxel_intensities, sv_partitioning, num_supervoxels)
    # print(voxel_labels.shape, sv_partitioning.shape)
    sv_labels = ndimage.labeled_comprehension(voxel_labels, labels=sv_partitioning[:, :, 0], func=mode, index=range(
        0, num_supervoxels), out_dtype='int32', default=-1.0)

    # centroid=center of mass where the mass is equally distributed *taps forehead*
    sv_centroids = np.array(ndimage.center_of_mass(
        np.ones(sv_partitioning.shape), sv_partitioning, range(0, num_supervoxels)))

    new_regions = np.zeros(num_supervoxels, dtype=np.int16)
    n_nodes = 0
    for i in range(num_supervoxels):
        new_regions[i] = n_nodes
        n_nodes += 1
    new_region_img = new_regions[sv_partitioning]

    return new_region_img, sv_feats, sv_centroids, sv_labels


def get_sv_summary_for_modality(modality_intensities, sv_partitioning, n_svs):
    # print(modality_intensities.shape)
    # print(sv_partitioning.shape)
    sv_feats = 0
    # print(modality_intensities.shape[2])
    for i in range(modality_intensities.shape[2]):
        temp_feats = ndimage.labeled_comprehension(
            modality_intensities[:, :, i], labels=sv_partitioning[:, :, i], func=get_quantiles, index=range(n_svs), out_dtype='object', default=-1.0)
        # temp_feats = np.stack(sv_feats, axis=0)
        # print(temp_feats.shape)
        if i == 0:
            sv_feats = np.stack(temp_feats, axis=0)
        else:
            sv_feats = np.concatenate(
                (sv_feats, np.stack(temp_feats, axis=0)), axis=1)
        # print(sv_feats.shape)
    # exit()
    return sv_feats


def find_adjacent_nodes(regionImg, n_nodes, as_mat=False):

    # first replace -1 with the next largest region number (i.e. the current number of nodes because 0 indexed) to ease computation
    tmp = np.zeros((n_nodes, n_nodes), bool)

    if(len(regionImg.shape) == 3):
        # check the vertical adjacency
        a, b = regionImg[:-1, :, :], regionImg[1:, :, :]
        tmp[a[a != b], b[a != b]] = True
        # check the horizontal adjacency
        a, b = regionImg[:, :-1, :], regionImg[:, 1:, :]
        tmp[a[a != b], b[a != b]] = True
        # check the depth adjacency
        a, b = regionImg[:, :, :-1], regionImg[:, :, 1:]
        tmp[a[a != b], b[a != b]] = True
    # 2D case
    else:
        a, b = regionImg[:-1, :], regionImg[1:, :]
        tmp[a[a != b], b[a != b]] = True
        a, b = regionImg[:, :-1], regionImg[:, 1:]
        tmp[a[a != b], b[a != b]] = True

    # register adjacency in both directions (up, down) and (left,right)
    adj_mat = (tmp | tmp.T)

    np.fill_diagonal(adj_mat, True)
    # return results as adjacency matrix
    if(as_mat):
        return adj_mat
    return np.where(adj_mat)


def img2graph(voxel_intensities, voxel_labels, approx_num_nodes=15000, boxiness=0.5):
    labels_provided = True if voxel_labels is not None else False
    slic_partitioning = slic(voxel_intensities.astype(np.float64), n_segments=approx_num_nodes,
                             sigma=1, compactness=boxiness, convert2lab=False, start_label=0)
    if len(slic_partitioning.shape) < 3:
        temp = np.zeros(
            (slic_partitioning.shape[0], slic_partitioning.shape[1], 3))
        temp[:, :, 0] = slic_partitioning
        temp[:, :, 1] = slic_partitioning
        temp[:, :, 2] = slic_partitioning
        slic_partitioning_updated = temp.astype(np.int16)
        # temp = np.zeros(
        #     (slic_partitioning.shape[0], slic_partitioning.shape[1], 4))
        # temp[:, :, 0] = slic_partitioning
        # temp[:, :, 1] = slic_partitioning
        # temp[:, :, 2] = slic_partitioning
        # temp[:, :, 3] = slic_partitioning
        # slic_partitioning_updated = temp.astype(np.int16)

    num_supervoxels = np.amax(slic_partitioning_updated)+1
#     print("Number of supervoxels generated by SLIC: ", num_supervoxels)

    if(not labels_provided):
        voxel_labels = np.zeros(voxel_intensities.shape[:3], dtype=np.int16)

    updated_partitioning, sv_feats, sv_centroids, sv_labels = determine_nodes_and_features(
        np.copy(slic_partitioning_updated), voxel_intensities, voxel_labels, num_supervoxels)

    graph_adjacency = find_adjacent_nodes(
        updated_partitioning, len(sv_labels), as_mat=True)

    nx_graph = nx.from_numpy_matrix(graph_adjacency)
    for n in nx_graph.nodes:
        if(labels_provided):
            label = int(sv_labels[n])
            nx_graph.nodes[n]["label"] = label
        features = list(sv_feats[n])
        nx_graph.nodes[n]["features"] = features

    return nx_graph, sv_feats, updated_partitioning, slic_partitioning
