import cv2
import numpy as np
import matplotlib.pyplot as plt
import ray

# Initialize Ray
ray.init()



# Load the deformed image
deformed_image = cv2.imread('deformed_image_parallel.jpg')
height, width, channels = deformed_image.shape

# Define force and points of application (same as deformation)
forces = [
    {'power': 500, 'position': np.array([width // 2, height // 2])},
    {'power': 300, 'position': np.array([width // 3, height // 3])},
    {'power': 200, 'position': np.array([2 * width // 3, 2 * height // 3])}
]

# Kelvinlet parameters
mu = 1.0  # Shear modulus
alpha = 1.0  # Elastic parameter
epsilon = 1e-6  # Regularization to avoid division by zero

# Kelvinlet function for displacement calculation
def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
    return u

# Ray task to process rows in parallel
@ray.remote
def recover_rows(start_row, end_row, forces, width, height, deformed_image):
    recovered_chunk = np.zeros_like(deformed_image[start_row:end_row])

    for y in range(start_row, end_row):
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

            # Assign the pixel from the deformed image to the recovered chunk
            recovered_chunk[y - start_row, x] = deformed_image[orig_y, orig_x]

    return recovered_chunk

# Split the work into chunks (parallel processing)
num_workers = 8  # Number of parallel workers
chunk_size = height // num_workers
futures = []

for i in range(num_workers):
    start_row = i * chunk_size
    end_row = (i + 1) * chunk_size if i < num_workers - 1 else height
    futures.append(recover_rows.remote(start_row, end_row, forces, width, height, deformed_image))

# Collect the results
results = ray.get(futures)

# Combine the results back into the final recovered image
recovered_image = np.vstack(results)

# Show the deformed and recovered images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB))
plt.title('Deformed Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB))
plt.title('Recovered Image (Parallelized)')

plt.show()

# Shutdown Ray
ray.shutdown()
