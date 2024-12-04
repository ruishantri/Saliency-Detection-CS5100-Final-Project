import cv2
import numpy as np

def detect_movement(video_path, output_path):
    # Initialize the background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the VideoWriter to save the output video as MP4
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = backSub.apply(frame)

        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # Draw bounding box around the largest moving object
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Green bounding box

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Output video saved at {output_path}")

# Example usage
video_file = "dogs_1.mp4"  # Path to your input video
output_video = "dogs_1_bs_red.mp4"  # Path for the output video
detect_movement(video_file, output_path=output_video)
