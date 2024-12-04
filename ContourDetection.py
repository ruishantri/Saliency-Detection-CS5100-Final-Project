import cv2
import numpy as np
import time
import os

def resize_with_aspect_ratio(image, target_size=(400, 400), padding_color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    delta_width = target_width - new_width
    delta_height = target_height - new_height
    top, bottom = delta_height // 2, delta_height - delta_height // 2
    left, right = delta_width // 2, delta_width - delta_width // 2
    resized_with_padding = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
    )
    return resized_with_padding

def create_thumbnail_with_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x, centroid_y = 0, 0
    centroid = (centroid_x, centroid_y)

    x, y, w, h = cv2.boundingRect(largest_contour)
    image_with_bbox = image.copy()
    cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.circle(image_with_bbox, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

    print(f"Centroid of the bounding box with respect to the original image: {centroid}")

    return image_with_bbox, centroid

# Process all .png files in the given folder
input_folder = "dogs_1_frames"
output_folder = "dogs_1_frames_contour_red"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

sum_time = 0
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or could not be loaded: {image_path}")
        continue

    start_time = time.time()
    image_with_bbox, centroid = create_thumbnail_with_contours(image)
    end_time = time.time()

    print(f"Execution Time for {image_file}: {end_time - start_time:.2f} seconds")
    sum_time += (end_time - start_time)

    output_path = os.path.join(output_folder, f"bbox_{image_file}")
    cv2.imwrite(output_path, image_with_bbox)
    print(f"Processed image saved at {output_path}")

print(f"Average processing time per image: {sum_time / len(image_files):.2f} seconds")
