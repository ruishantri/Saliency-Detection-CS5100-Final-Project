import cv2
import numpy as np

def detect_movement_with_combined_approach(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the VideoWriter to save the output video as MP4
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        fg_mask = back_sub.apply(frame)

        # Frame difference
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh_diff = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)

        # Combine the results: use bitwise AND to combine both background subtraction and frame differencing
        combined_mask = cv2.bitwise_and(fg_mask, thresh_diff)

        # Find contours of the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # Update previous frame for the next iteration
        prev_gray = gray

    # Release resources
    cap.release()
    out.release()

    print(f"Output video saved at {output_path}")

# Example usage
video_file = "dogs_1.mp4"  # Path to your input video
output_video = "dogs_1_bs_fd_red.mp4"  # Path for the output video
detect_movement_with_combined_approach(video_file, output_path=output_video)
