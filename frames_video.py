import cv2
import os
from natsort import natsorted  # For natural sorting (e.g., frame_1.png, frame_2.png)

def frames_to_video(input_folder, output_video_path, frame_rate=30):
    """
    Converts a sequence of frames into a video.

    Parameters:
        input_folder (str): Path to the folder containing the frames.
        output_video_path (str): Path to save the output video.
        frame_rate (int): Frame rate of the output video.
    """
    # Get a sorted list of frame file names
    frame_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))]
    frame_files = natsorted(frame_files)  # Natural sorting for frame_1, frame_2, ...

    if not frame_files:
        raise ValueError("No valid frame files found in the input folder.")

    # Read the first frame to get the dimensions
    first_frame_path = os.path.join(input_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Could not read the first frame: {first_frame_path}")
    
    height, width, layers = first_frame.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Skipping unreadable frame: {frame_path}")
            continue
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved at: {output_video_path}")

# Example usage
input_folder = "dogs_1_frames_contour_red"  # Replace with the path to your frames folder
output_video_path = "output_video_dogs_1_contour_red.mp4"  # Path to save the video
frame_rate = 30  # Adjust as needed

frames_to_video(input_folder, output_video_path, frame_rate)



