import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paths to your video files
video_path1 = '../output/videos/glass_dis__.avi'  # First video path
video_path2 = '../output/videos/glass_inv__.avi'  # Second video path

# Open video files
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# Get video properties
fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
fps = min(fps1, fps2)  # Use the minimum FPS for synchronization

# Prepare the figure
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create side-by-side subplots

def update(frame):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        return [], []
    frame1 = cv2.resize(frame1, (512, 512))  # Resize to 640x360
    frame2 = cv2.resize(frame2, (512, 512))

    # Convert BGR (OpenCV format) to RGB
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Clear previous frames
    ax[0].cla()
    ax[1].cla()

    # Display the frames
    ax[0].imshow(frame1)
    ax[1].imshow(frame2)

    # Set titles
    ax[0].set_title("Video 1")
    ax[1].set_title("Video 2")

    # Hide axes
    ax[0].axis('off')
    ax[1].axis('off')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=None, interval=1000/fps)

# Show the plot
plt.show()

# Release video capture objects
cap1.release()
cap2.release()
