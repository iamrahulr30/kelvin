import cv2
import os

frames_folder = '../parallel/outframes'  
output_video_path = '../output/videos/glass_dis1.avi'  
fps = 30  

frame_files = os.listdir(frames_folder)

frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
# print(frame_files)

if len(frame_files) == 0:
    print("No frames found in the specified folder.")
else:
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at: {output_video_path}")
