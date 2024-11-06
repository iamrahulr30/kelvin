import cv2
import os
import numpy as np

# Paths
output_folder = '../parallel/outframes/'
recovered_folder = '../parallel/inv_imgs/'
os.makedirs(recovered_folder, exist_ok=True)

# Kelvinlet parameters
mu = 0.5
alpha = -0.4
epsilon = 1e-6  # divide-by-zero errors

def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r, axis=1) + epsilon  
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / r_norm**3)).reshape(-1, 1) * F
    return u

def kelvinlet_inversion(deformed_image, forces):
    height, width, _ = deformed_image.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords = np.stack((x_indices, y_indices), axis=-1).reshape(-1, 2)

    displacement = np.zeros(coords.shape)
    for force in forces:
        r = coords - force['position']
        displacement += kelvinlet_displacement(r, force['power'])

    A = np.eye(2)
    A_inv = np.linalg.inv(A)  
    original_coords = (A_inv @ (coords - displacement).T).T  
    original_coords = np.clip(original_coords, [0, 0], [width - 1, height - 1]).astype(int)
    recovered_image = deformed_image[original_coords[:, 1], original_coords[:, 0]].reshape(height, width, -1)

    return recovered_image

def process_frame(frame_no, power, y_position, deformed_image):
    forces = [{'power': power, 'position': np.array([260.0, y_position])}]
    recovered_image = kelvinlet_inversion(deformed_image, forces)

    recovered_filename = os.path.join(recovered_folder, f"inv_frame_{frame_no:04}.jpg")
    cv2.imwrite(recovered_filename, recovered_image)
    print(f"Recovered and saved {recovered_filename}")


with open('../parallel/dis.txt', 'r') as file:
    displacements = eval(file.readline().strip())  

all_frames = {
    frame_no: cv2.imread(f"{output_folder}/frame_{frame_no:04}.png")
    for frame_no, _, _ in displacements
}

all_frames = {k: v for k, v in all_frames.items() if v is not None}

for frame_no, power, y_position in displacements:
    if frame_no in all_frames:
        process_frame(frame_no, power, y_position, all_frames[frame_no])

print("All frames processed sequentially.")
