import cv2
import os
import numpy as np
import ray

# Initialize Ray
ray.init()

# Paths
output_folder = 'outframes/'
os.makedirs(output_folder, exist_ok=True)
recovered_folder = 'inv_imgs/'
os.makedirs(recovered_folder, exist_ok=True)

# Kelvinlet parameters
mu = .5
alpha = -.4
epsilon = 1

# Function to calculate Kelvinlet displacement
def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
    return u

# Remote function to process frames
@ray.remote
def process_frame(deformed_image, frame_no, power, y_position):
    height, width, _ = deformed_image.shape
    forces = [{'power': power, 'position': np.array([260.0, y_position])}]
    recovered_image = np.zeros_like(deformed_image)

    for y in range(height):
        for x in range(width):
            displacement = np.zeros(2)
            for force in forces:
                r = np.array([x, y]) - force['position']  # Relative position vector
                u = kelvinlet_displacement(r, force['power'])  # Displacement
                displacement += u  # Accumulate displacement

            # Set up the A matrix
            A = np.eye(2)  # 2D identity matrix
            b = np.array([x, y]) - displacement  # Reverse the displacement
            original_coords = np.linalg.solve(A, b)
            orig_x, orig_y = np.clip(original_coords, [0, 0], [width - 1, height - 1]).astype(int)

            recovered_image[y, x] = deformed_image[orig_y, orig_x]

    return recovered_image

# Remote function to save images
@ray.remote
def save_image(image, filename):
    cv2.imwrite(filename, image)

# Remote function to read images
@ray.remote
def read_image(filename):
    image = cv2.imread(filename)
    if image is None:
        print(f"Warning: {filename} could not be found or opened.")
    return image

# Load the displacement values from file
with open('dis.txt', 'r') as file:
    line = file.readline().strip()
    displacements = eval(line)  # List of (frame_no, power, y_position)

print(displacements)
# Read images in parallel and retrieve results
read_tasks = [
    read_image.remote(f"{output_folder}/frame_{frame_no + 1:04}.png") 
    for frame_no, _, _ in displacements
]
read_results = ray.get(read_tasks)

# Process frames in parallel
process_tasks = [
    process_frame.remote(deformed_image, frame_no + 1, power, y_position)
    for (frame_no, power, y_position), deformed_image in zip(displacements, read_results)
    if deformed_image is not None
]

# Retrieve processed images
processed_results = ray.get(process_tasks)

# Save processed images in parallel with correct frame numbers
save_tasks = [
    save_image.remote(processed_image, os.path.join(recovered_folder, f"frame_{frame_no + 1:04}.jpg"))
    for (frame_no, _, _), processed_image in zip(displacements, processed_results)
    if processed_image is not None
]

# Wait for all save tasks to complete
ray.get(save_tasks)

# Shutdown Ray
ray.shutdown()
