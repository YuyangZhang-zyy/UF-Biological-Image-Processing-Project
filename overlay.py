import os
import re
import tifffile
import numpy as np
import xml.etree.ElementTree as ET
import cv2

def overlay_segmented_image(original_image, segmented_image, annotation):
    x, y, width, height = annotation
    # Ensure the annotation coordinates and dimensions are within the segmented image bounds
    x = max(0, int(x))
    y = max(0, int(y))
    width = min(segmented_image.shape[1], int(width))
    height = min(segmented_image.shape[0], int(height))
    # Check if width and height are valid for resizing
    if width > 0 and height > 0:
        # Resize the segmented image to match the size of the annotation
        resized_segmented_image = cv2.resize(segmented_image, (width, height))
        # Overlay the resized segmented image onto the original image
        original_image[y:y+height, x:x+width] = resized_segmented_image
    return original_image

# Load the original image
original_image_path = 'P44F_PAS.svs'
original_image = tifffile.imread(original_image_path)

# Parse the XML file to extract annotations
xml_file = 'P44F_PAS.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

annotations = []
for region in root.findall('.//Region'):
    vertices = region.findall('.//Vertices/Vertex')
    x_coords = [float(vertex.get('X')) for vertex in vertices]
    y_coords = [float(vertex.get('Y')) for vertex in vertices]
    x = min(x_coords)
    y = min(y_coords)
    width = max(x_coords) - x
    height = max(y_coords) - y
    annotations.append((x, y, width, height))

# Sort segmented image filenames to match the order of annotations
segmented_images = sorted([filename for filename in os.listdir('result') if filename.endswith('_result.tif')], key=lambda x: int(re.search(r'\d+', x).group()))

# Overlay segmented images onto the original image based on annotations
final_image = original_image.copy()

for segmented_image_filename, annotation in zip(segmented_images, annotations):
    segmented_image_path = os.path.join('result', segmented_image_filename)
    segmented_image = tifffile.imread(segmented_image_path)
    final_image = overlay_segmented_image(final_image, segmented_image, annotation)

# Save the final result
final_result_path = os.path.join('result', 'final_result.tif')
tifffile.imsave(final_result_path, final_image)