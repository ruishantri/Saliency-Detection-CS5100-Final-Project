import cv2
import numpy as np
import time
import os

def resize_with_aspect_ratio(image, target_size=(400, 400), padding_color=(0, 0, 0)):
    """
    Resize the image to fit within the target size while preserving the aspect ratio.
    Pads with the specified color to maintain square dimensions.
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factor to fit within the target size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding to make the final image square
    delta_width = target_width - new_width
    delta_height = target_height - new_height
    top, bottom = delta_height // 2, delta_height - delta_height // 2
    left, right = delta_width // 2, delta_width - delta_width // 2

    # Add padding
    resized_with_padding = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
    )

    return resized_with_padding

def create_thumbnail(image):
    """
    Automatically segment the foreground and draw a bounding box.
    Calculate the centroid of the bounding box with respect to the original image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's threshold to separate background and probable foreground
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to remove noise and improve the binary mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilated mask)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (distance transform and thresholding)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region (subtract sure foreground from sure background)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    markers = cv2.connectedComponents(sure_fg)[1]

    # Add one to all labels so the background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with 0
    markers[unknown == 255] = 0

    # Apply the watershed algorithm
    markers = cv2.watershed(image, markers)

    # Separate foreground based on markers
    mask = markers > 1  # Markers > 1 represent the foreground

    # Find bounding box of the foreground
    coords = np.column_stack(np.where(mask))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Calculate the centroid of the bounding box
    centroid_x = (x_min + x_max) // 2
    centroid_y = (y_min + y_max) // 2
    centroid = (centroid_x, centroid_y)

    # Draw the bounding box and centroid on the original image
    image_with_bbox = image.copy()
    cv2.rectangle(image_with_bbox, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)  # Green box
    # print(f"Centroid of the bounding box with respect to the original image: {centroid}")

    return image_with_bbox, centroid

# Process all .jpg files in the given folder
input_folder = "dogs_1_frames"
output_folder = "dogs_1_frames_water_red"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

sum_time = 0
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or could not be loaded: {image_path}")
        continue

    # Process the image
    start_time = time.time()
    image_with_bbox, centroid = create_thumbnail(image)
    end_time = time.time()

    print(f"Execution Time for {image_file}: {end_time - start_time:.2f} seconds")
    sum_time += (end_time - start_time)

    # Save the processed image with bounding box
    output_path = os.path.join(output_folder, f"bbox_{image_file}")
    cv2.imwrite(output_path, image_with_bbox)
    print(f"Processed image saved at {output_path}")

# Print average processing time
if image_files:
    print(f"Average processing time per image: {sum_time / len(image_files):.2f} seconds")
else:
    print("No .jpg images found in the input folder.")
