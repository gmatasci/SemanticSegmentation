import numpy as np

# Convert a 2D tensor of class labels (starting at 1) to a 3D one-hot version (as many layers in depth as classes)
def map_2_one_hot(map, nr_classes):
    map_flat = map.reshape(map.shape[0]*map.shape[1])
    map_flat_one_hot = np.zeros((map_flat.size, nr_classes))
    map_flat_one_hot[np.arange(map_flat.size), map_flat-1] = 1
    return map_flat_one_hot.reshape(map.shape[0], map.shape[1], nr_classes)

# Sort patch names numerically first by area then by patch number
def sort_patch_names(patches_names, ID_list):
    patches_names_sorted = []
    for ID in ID_list:
        patches_names_area = [patch_name for patch_name in patches_names if '_area%d_' % ID in patch_name]
        patch_ID_ext = [name.split('_p')[1] for name in patches_names_area]
        patch_ID = [int(name.split('.png')[0]) for name in patch_ID_ext]
        sorting_idx = np.argsort(patch_ID)
        patches_names_sorted.extend([patches_names_area[i] for i in sorting_idx])
    return patches_names_sorted