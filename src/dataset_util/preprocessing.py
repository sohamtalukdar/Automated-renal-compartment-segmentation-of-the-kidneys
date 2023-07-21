import numpy as np
import pandas as pd
import os
import nibabel as nib
from nibabel.processing import conform
from sklearn.model_selection import train_test_split
from skimage.transform import resize
# All the preprocessing functions you need, including the load_data function.

def rescale(data, black=None, white=None):
    if black is None:
        black = np.mean(data) - 0.5 * np.std(data)
        if black < data.min():
            black = data.min()
    if white is None:
        white = np.mean(data) + 4 * np.std(data)
        if white > data.max():
            white = data.max()
    data = np.clip(data, black, white) - black
    data = data / (white - black)
    return data

def normalise_image(image, affine):
    image = rescale(image)
    img = nib.Nifti1Image(image, affine)
    zoom = img.header.get_zooms()
    img = conform(img, out_shape=(240, 240, img.shape[-1]),
                  voxel_size=(1.458, 1.458, zoom[-1] * 0.998),
                  orientation='LIP')
    image = img.get_fdata()
    image = resize(image, (256, 256, image.shape[2]))
    zoom = img.header.get_zooms()
    return image

def normalise_mask(mask, affine):
    img = nib.Nifti1Image(mask, affine)
    img = conform(img, out_shape=(240, 240, img.shape[-1]),
                  voxel_size=(1.458, 1.458, img.header.get_zooms()[-1] *
                              0.998),
                  orientation='LIP')
    mask = img.get_fdata()
    mask = resize(mask, (256, 256, mask.shape[2]))
    mask = np.round(mask)
    return mask

# Load data function
# ... (your other functions)

# Load data function
def load_data(dir, save_dir=None):
    ground_truths = []
    base_scans = []

    for f in sorted(os.listdir(dir + 'GroundTruth/')):
        if not f.startswith('.'):  # Skip hidden files like .DS_Store
            image = nib.load(os.path.join(dir, 'GroundTruth', f))
            ground_truths.append(image)

    for f in sorted(os.listdir(dir + 'Masks_T1/')):
        if not f.startswith('.'):  # Skip hidden files like .DS_Store
            image = nib.load(os.path.join(dir, 'Masks_T1', f))
            base_scans.append(image)

    gt_arrays = []
    gt_affine = []
    base_arrays = []
    base_affine = []

    for img in ground_truths:
        affine = img.affine
        gt_voxels = img.get_fdata()
        normalised = normalise_mask(gt_voxels, affine)
        gt_arrays.append(normalised)

    for img in base_scans:
        affine = img.affine
        base_voxels = img.get_fdata()
        normalised = normalise_image(base_voxels, affine)
        base_arrays.append(base_voxels)

    x_series = pd.Series(base_arrays)
    y_series = pd.Series(gt_arrays)

    x_data = tf_arrays(x_series)
    y_data = tf_arrays(y_series)

    if save_dir:
            # Save the processed data to a Parquet file
            processed_data = pd.DataFrame({'base_arrays': base_arrays, 'gt_arrays': gt_arrays})
            processed_data.to_parquet(os.path.join(save_dir, 'processed_data.parquet'))

    return train_test_split(x_data, y_data, test_size=0.1)


def tf_arrays(series):
    data = np.concatenate(series, 2)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    data = np.expand_dims(data, 3)
    return data
