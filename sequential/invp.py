import cv2
import os
import numpy as np

# Paths
output_folder = '../parallel/outframes/'
recovered_folder = '../parallel/inv_imgs/'
os.makedirs(recovered_folder, exist_ok=True)

# Kelvinlet parameters (should match the deformation parameters exactly)
mu = .5#1
alpha = -.4 #-2
epsilon = 1

# Function to calculate Kelvinlet displacement
def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
    return u

# Function to reverse displacement on an image
def reverse_displacement(deformed_image, forces):
    height, width, _ = deformed_image.shape
    recovered_image = np.zeros_like(deformed_image)

    for y in range(height):
        for x in range(width):
            displacement = np.zeros(2)

            # Calculate total displacement for this pixel
            for force in forces:
                r = np.array([x, y]) - force['position']  # Relative position vector
                u = kelvinlet_displacement(r, force['power'])  # Displacement
                displacement += u  # Accumulate displacement from all forces

            # Set up the A matrix (2x2 identity for linear deformation approximation)
            A = np.eye(2)  # 2D identity matrix

            # Set up the b vector (displaced position)
            b = np.array([x, y]) - displacement  # Reverse the displacement

            # Solve for the original position (A * original_position = displaced_position)
            original_coords = np.linalg.solve(A, b)

            # Clip the coordinates to be within the image bounds
            orig_x, orig_y = np.clip(original_coords, [0, 0], [width - 1, height - 1]).astype(int)

            # Assign the pixel from the deformed image to the recovered image
            recovered_image[y, x] = deformed_image[orig_y, orig_x]

    return recovered_image

# Load the displacement values from file
with open('../parallel/dis.txt', 'r') as file:
    line = file.readline().strip()
    displacements = eval(line)  # List of (frame_no, power, y_position)

# Process each deformed frame
for frame_no, power, y_position in displacements:
    frame_no += 1
    filename = f"{output_folder}/frame_{frame_no:04}.png"  # Zero-padded frame number
    deformed_image = cv2.imread(filename)

    if deformed_image is None:
        print(f"Warning: {filename} could not be found or opened.")
        continue

    # Forces applied to the image (must match original deformation)
    forces = [{'power': power, 'position': np.array([260.0, y_position])}]

    # Reverse the displacement to recover the original image
    recovered_image = reverse_displacement(deformed_image, forces)

    # Save the recovered image
    recovered_filename = os.path.join(recovered_folder, f"inv_frame_{frame_no:04}.jpg")
    cv2.imwrite(recovered_filename, recovered_image)
    print(f"Recovered and saved {recovered_filename}")
