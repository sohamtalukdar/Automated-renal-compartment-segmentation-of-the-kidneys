import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to 
import scipy.ndimage as ndi
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cv2

# Load image data
def load_nifti_data(file_path):
    img = nib.load(file_path)
    voxels = img.get_fdata()
    return voxels

t1_voxels = load_nifti_data('Dataset/Data/Data/S01V02_MPRAGE.nii.gz')
t2_voxels = load_nifti_data('Dataset/Data/Data/S01V02_T1W.nii.gz')
t3_voxels = load_nifti_data('Dataset\Data\Data\S01V02_T2W.nii.gz')

# Plot a single slice
def plot_slice(voxels, slice_idx, title):
    plt.imshow(ndi.rotate(voxels[:, :, slice_idx], 90), cmap='gray_r')
    plt.title(title)
    plt.axis(False)
    plt.show()

# Display central axial slices for MPRAGE, T1W, and T2W
central_slice_idx = t1_voxels.shape[2] // 2
plot_slice(t1_voxels, central_slice_idx, 'MPRAGE')
plot_slice(t2_voxels, central_slice_idx, 'T1W')
plot_slice(t3_voxels, central_slice_idx, 'T2W')

# Get voxel size information
t1_hdr = nib.load('Dataset\Data\Data\S01V02_MPRAGE.nii.gz').header
zoom = t1_hdr.get_zooms()
print('The voxels of this image are {0:.1f} x {1:.1f} x {2:.1f} mm'.format(*zoom))

# Load and plot the mask
mask_voxels = load_nifti_data('Dataset/Data/Data/S01V02_T2W_mask.nii.gz') > 0.5
plot_slice(mask_voxels, central_slice_idx, 'Mask')

# Apply mask and plot masked kidney
masked_t1_voxels = mask_voxels * t1_voxels
plot_slice(masked_t1_voxels, central_slice_idx, 'Masked kidney')
