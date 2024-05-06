import os
import tifffile
import numpy as np
from skimage.color import rgb2hed
from skimage import filters, morphology, segmentation, feature
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.measure import regionprops
from sklearn.cluster import KMeans
import pandas as pd
import xml.etree.ElementTree as ET

OMP_NUM_THREADS=1

# Function to segment nuclei and overlay them on the original color image
def segment_nuclei(image_path, save_path):
    # Load the original color image
    color_image = tifffile.imread(image_path)
    
    cropped_glom = rgb2hed(color_image)[:,:,0]

    # Thresholding
    thresh = filters.threshold_otsu(cropped_glom)
    binary = cropped_glom > thresh

    # Morphological operations
    binary_mask = morphology.binary_opening(binary, morphology.disk(4))

    # Generate markers
    distance = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, footprint=np.ones((28, 28)), labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary_mask)

    # Find boundaries of watershed elements
    boundaries = find_boundaries(labels)

    # Initialize lists to store feature values
    mean_intensities = []
    variances = []
    min_intensities = []
    max_intensities = []
    areas = []
    perimeters = []
    eccentricities = []

    # Iterate over each labeled region
    for label in range(1, labels.max() + 1):  # Iterate over each label excluding background
        # Mask out the region in the original image
        region_mask = labels == label
        region_pixels = cropped_glom[region_mask]
    
        # Compute mean intensity of the region
        mean_intensity = np.mean(region_pixels)
        mean_intensities.append(mean_intensity)
    
        # Compute variance of intensity in the region
        variance = np.var(region_pixels)
        variances.append(variance)
    
        # Compute minimum intensity in the region
        min_intensity = np.min(region_pixels)
        min_intensities.append(min_intensity)
    
        # Compute maximum intensity in the region
        max_intensity = np.max(region_pixels)
        max_intensities.append(max_intensity)

    # Z-score normalization
    mean_intensities_normalized = (mean_intensities - np.mean(mean_intensities)) / np.std(mean_intensities)
    variances_normalized = (variances - np.mean(variances)) / np.std(variances)
    min_intensities_normalized = (min_intensities - np.mean(min_intensities)) / np.std(min_intensities)
    max_intensities_normalized = (max_intensities - np.mean(max_intensities)) / np.std(max_intensities)

    # Get region properties for each labeled region (nucleus)
    props = regionprops(labels)

    # # Iterate over each labeled region
    # for region in props:
    #     # Compute area of the region
    #     area = region.area
    #     areas.append(area)
    
    #     # Compute perimeter of the region
    #     perimeter = region.perimeter
    #     perimeters.append(perimeter)
    
    #     # Compute eccentricity of the region
    #     eccentricity = region.eccentricity
    #     eccentricities.append(eccentricity)
    
    # # Z-score normalize the morphological features
    # areas_normalized = (areas - np.mean(areas)) / np.std(areas)
    # perimeters_normalized = (perimeters - np.mean(perimeters)) / np.std(perimeters)
    # eccentricities_normalized = (eccentricities - np.mean(eccentricities)) / np.std(eccentricities)


    # Combine morphological features with the existing feature matrix
    feature_matrix = np.column_stack((mean_intensities_normalized, variances_normalized, 
                                  min_intensities_normalized, max_intensities_normalized,
                                  # areas_normalized, perimeters_normalized, eccentricities_normalized
                                  ))

    # Define the number of clusters
    num_clusters = 2  # Adjust this according to your data

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    cluster_centers = kmeans.cluster_centers_

    # Add cluster labels to props for visualization or further analysis
    for i, region in enumerate(props):
        region.cluster_label = cluster_labels[i]

    # Assign colors based on cluster characteristics
    cluster_colors = []
    for i in range(num_clusters):
        # Assign colors based on feature values
        if cluster_centers[i, 0] > 0:  # Example: Mean intensity is high
            color = 'b'  # Blue
        else:
            color = 'g'  # Green
        cluster_colors.append(color)

    # # Perform k-means clustering
    # kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    # cluster_labels = kmeans.fit_predict(feature_matrix)

    # # Add cluster labels to props for visualization or further analysis
    # for i, region in enumerate(props):
    #     region.cluster_label = cluster_labels[i]

    # # Define colors for each cluster label
    # cluster_colors = ['b', 'g', 'y', 'c', 'm']  # Add more colors if needed

    # Overlay nuclei clusters with different colors on a blank image
    overlayed_image = color_image.copy()

    for region in props:
        cluster_label = region.cluster_label
        if cluster_label < len(cluster_colors):
            mask = labels == region.label
            color = np.array(matplotlib.colors.to_rgb(cluster_colors[cluster_label])) * 255
            overlayed_image[mask] = color
    
    # Save the resulting image
    filename = os.path.basename(image_path)
    result_filename = filename.split('.')[0] + '_result.tif'
    result_path = os.path.join(save_path, result_filename)
    tifffile.imsave(result_path, overlayed_image)

# Path to the folder containing glom images
folder_path = 'gloms'

# Create a folder named 'result' to save the output images
output_folder = 'result'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Test all images and generate image to a file named 'result' 
# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image_path = os.path.join(folder_path, filename)
        
        # Segment nuclei and overlay them on the original color image
        segment_nuclei(image_path, output_folder)