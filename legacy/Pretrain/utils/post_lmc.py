import numpy as np
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import elf.segmentation.watershed as ws
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
from elf.segmentation.features import *
from elf.segmentation.learning import *
from elf.segmentation.mutex_watershed import mutex_watershed
from elf.parallel.relabel import relabel_consecutive
from nifty import tools as ntools
import nifty.graph.rag as nrag
import os
import time
from tqdm import tqdm
import numpy as np
import joblib
import imageio
from skimage.metrics import variation_of_information, adapted_rand_error
from multiprocessing import Pool, Lock

def post_lmc(affs):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
        wsz += offset
        offset += max_id
        watershed[z] = wsz
    rag = feats.compute_rag(watershed)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=0.25)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation

def post_lmc_lh(affs, beta):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
        wsz += offset
        offset += max_id
        watershed[z] = wsz
    rag = feats.compute_rag(watershed)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation

def post_mc_b(boundary_input, beta=0.25):
    boundary_input = 1 - boundary_input 
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=0.25, sigma_seeds=2.0)
        wsz += offset
        offset += max_id
        watershed[z] = wsz
    rag = feats.compute_rag(watershed)
    costs = compute_boundary_features(rag, boundary_input, min_value=0, max_value=1)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)

    return segmentation