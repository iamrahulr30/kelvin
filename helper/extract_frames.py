import cv2
import os

def extract_frames(video_path, output_folder="../data/frames_main", target_size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_number = 1

    while cap.isOpened()  :
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the target size (512x512)
        frame_resized = cv2.resize(frame, target_size)

        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        
        cv2.imwrite(frame_filename, frame_resized)
        print(f"Saved {frame_filename}")
        frame_number += 1

    cap.release()
    print("All frames have been extracted and saved.")

# Specify the video path
video_path = '../data/videos/glass2.mp4'
extract_frames(video_path)
