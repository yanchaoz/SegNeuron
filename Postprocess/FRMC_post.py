from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
import numpy as np
import imageio


def post_mc(affs, beta=0.25):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=0.25, sigma_seeds=2.0)
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


if __name__ == "__main__":
    aff_root = '/***/***'
    bound_root = '/***/***'
    gt_root = '/***/***'
    beta = 0.25

    gt_seg = imageio.volread(gt_root)
    gt_seg = np.uint32(gt_seg)

    boundary_input = imageio.volread(bound_root)
    boundary_input = np.array([boundary_input, boundary_input, boundary_input])
    affine = np.load(aff_root)
    affine = np.minimum(boundary_input, affine)

    pred_seg = post_mc(affine, beta)

    arand = adapted_rand_ref(gt_seg, pred_seg, ignore_labels=(0,))[0]
    voi_split, voi_merge = voi_ref(gt_seg, pred_seg, ignore_labels=(0,))

    voi_sum = voi_split + voi_merge
    print('voi_split:', voi_split, 'voi_merge:', voi_merge, 'voi:', voi_sum, 'arand', arand)
