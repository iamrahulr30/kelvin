import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_images(original_path, displaced_path, recovered_path):
    # Load images
    original_img = cv2.imread(original_path)
    displaced_img = cv2.imread(displaced_path)
    recovered_img = cv2.imread(recovered_path)
    
    # Convert from BGR to RGB for display
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    displaced_img = cv2.cvtColor(displaced_img, cv2.COLOR_BGR2RGB)
    recovered_img = cv2.cvtColor(recovered_img, cv2.COLOR_BGR2RGB)
    
    # Display images side by side
    plt.figure(figsize=(12, 8))
    
    # Plot recovered image (left)
    plt.subplot(1, 3, 1)
    plt.imshow(recovered_img)
    plt.title('Left: Recovered Image')
    plt.axis('off')
    
    # Plot displaced image (middle)
    plt.subplot(1, 3, 2)
    plt.imshow(displaced_img)
    plt.title('Middle: Displaced Image')
    plt.axis('off')
    
    # Plot original image (right)
    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.title('Right: Original Image')
    plt.axis('off')
    
    plt.suptitle("Comparison of Images", fontsize=16)
    plt.show()

# Usage:
org = "../data/frame_out/frame_0047.png"
dis = "../parallel/outframes/frame_0047.png"
inv = "../parallel/inv_imgs/inv_frame_0047.jpg"
compare_images(org , dis , inv)
