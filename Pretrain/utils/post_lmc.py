import elf.segmentation.features as feats
import elf.segmentation.multicut as mc
import elf.segmentation.watershed as ws
import numpy as np
from elf.segmentation.learning import compute_boundary_features


def post_lmc(affs):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    watershed = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(
            boundary_input[z], threshold=0.25, sigma_seeds=2.0
        )
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
    watershed = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(
            boundary_input[z], threshold=0.25, sigma_seeds=2.0
        )
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
    watershed = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(
            boundary_input[z], threshold=0.25, sigma_seeds=2.0
        )
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
