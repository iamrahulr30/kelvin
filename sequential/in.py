import cv2
import os
import numpy as np
import ray

# Initialize Ray
ray.init(num_cpus=4)

# Paths
output_folder = '../parallel/outframes/'
recovered_folder = '../parallel/inv_imgs/'
os.makedirs(recovered_folder, exist_ok=True)

mu = 0.5
alpha = -0.4
epsilon = 1

def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
    return u

def kelvinlet_inversion(deformed_image, forces):
    height, width, _ = deformed_image.shape
    recovered_image = np.zeros_like(deformed_image)

    for y in range(height):
        for x in range(width):
            displacement = np.zeros(2)

            for force in forces:
                r = np.array([x, y]) - force['position']
                u = kelvinlet_displacement(r, force['power'])
                displacement += u

            A = np.eye(2)
            b = np.array([x, y]) - displacement
            original_coords = np.clip(np.linalg.solve(A, b), [0, 0], [width - 1, height - 1]).astype(int)
            orig_x, orig_y = original_coords
            recovered_image[y, x] = deformed_image[orig_y, orig_x]

    return recovered_image

@ray.remote
def process_frame(frame_no, power, y_position, deformed_image):
    print(frame_no)
    forces = [{'power': power, 'position': np.array([260.0, y_position])}]
    recovered_image = kelvinlet_inversion(deformed_image, forces)

    recovered_filename = os.path.join(recovered_folder, f"inv_frame_{frame_no:04}.jpg")
    cv2.imwrite(recovered_filename, recovered_image)
    return recovered_filename

with open('../parallel/dis.txt', 'r') as file:
    line = file.readline().strip()
    displacements = eval(line)  # List of (frame_no, power, y_position)

all_frames = {}
for frame_no, _, _ in displacements:
    filename = f"{output_folder}/frame_{frame_no:04}.png"
    deformed_image = cv2.imread(filename)
    if deformed_image is None:
        print(f"Warning: {filename} could not be found or opened.")
    else:
        all_frames[frame_no] = deformed_image

tasks = [
    process_frame.remote(frame_no, power, y_position, all_frames[frame_no])
    for frame_no, power, y_position in displacements if frame_no in all_frames
]

# Get results
results = ray.get(tasks)

for result in results:
    if result is not None:
        print(f"Recovered and saved {result}")

# Shutdown Ray
ray.shutdown()
