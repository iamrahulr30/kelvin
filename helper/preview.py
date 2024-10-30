import cv2
import matplotlib.pyplot as plt
import os

def display_media(file_path, invert_x=False, invert_y=False, resize=False):
    if not os.path.isfile(file_path):
        print("File not found.")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in image_extensions:
        media = cv2.imread(file_path)
        if media is None:
            print("Failed to load the image.")
            return
    elif file_extension in video_extensions:
        cap = cv2.VideoCapture(file_path)
        ret, media = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame from video.")
            return
    else:
        print("Unsupported file format.")
        return
    
    if resize:
        media = cv2.resize(media, (512, 512))
    
    rgb_media = cv2.cvtColor(media, cv2.COLOR_BGR2RGB)
    
    plt.imshow(rgb_media)
    if invert_x:
        plt.gca().invert_xaxis()
    if invert_y:
        plt.gca().invert_yaxis()
    plt.axis('on')
    plt.show()


file = "../parallel/outframes/frame_0039.png"
display_media(file , invert_x= False, invert_y=False, resize=True)
