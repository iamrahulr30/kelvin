import cv2
import ray
import os
import time
import numpy as np

calc_time = []

no_workers = 4

frames_folder = '../data/frames_main/'

for i in range(100, 830, 100):
    ray.init(num_cpus=no_workers)

    start = time.time()

    start_frame = 0
    end_frame = i

    frame_files = os.listdir(frames_folder)[start_frame:end_frame]

    # Load and resize images to 512x512
    images = [
        cv2.resize(cv2.imread(os.path.join(frames_folder, file)), (512, 512)) for file in frame_files
    ]

    # Update total_iterations based on start_frame and end_frame
    total_iterations = end_frame - start_frame  # Total frames being processed
    midpoint = total_iterations // 2  # Calculate midpoint

    initial_y = 260
    final_y = 250
    initial_power = 3000
    max_power = 4000

    mu = 1.0
    alpha = -2
    epsilon = 1e-5  # Avoid division by zero

    def kelvinlet_displacement(r, F):
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True) + epsilon
        u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
        return np.nan_to_num(u)  # Replace NaN with 0

    @ray.remote
    def process_frame(image, iter_index, power, y_position):
        height, width, _ = image.shape
        
        x_position = 260.0

        # Print debugging information
        # print(iter_index, power, y_position)

        forces = [{'power': power, 'position': np.array([x_position, y_position])}]

        # Create meshgrid for pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Initialize displacement array
        displacement = np.zeros((height, width, 2))

        # Calculate displacement for each force
        for force in forces:
            r = np.stack([x_coords - force['position'][0], y_coords - force['position'][1]], axis=-1)
            u = kelvinlet_displacement(r, force['power'])
            displacement += u  # Accumulate displacement

        # Calculate new coordinates and ensure they stay within bounds
        new_x = np.clip((x_coords + displacement[..., 0]).astype(np.float32), 0, width - 1)
        new_y = np.clip((y_coords + displacement[..., 1]).astype(np.float32), 0, height - 1)

        # Convert to integers and handle invalid values
        new_x = np.clip(new_x.astype(int), 0, width - 1)
        new_y = np.clip(new_y.astype(int), 0, height - 1)

        # Create the deformed image by mapping pixels to new coordinates
        deformed_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                deformed_image[i, j] = image[new_y[i, j], new_x[i, j]]

        return deformed_image

    @ray.remote
    def save_image(image, filename):
        cv2.imwrite(filename, image)
        # print(f"Saved {filename}")

    # Pre-compute power values
    power_values = np.zeros(total_iterations)
    power_values[:midpoint] = np.linspace(initial_power, max_power, midpoint)
    power_values[midpoint:] = np.linspace(max_power, 0, total_iterations - midpoint)

    # Pre-compute y_position values
    y_positions = np.linspace(initial_y, final_y, total_iterations)

    # Create Ray tasks for processing frames
    tasks = [
        process_frame.remote(images[i], start_frame + i, power_values[i], y_positions[i])
        for i in range(len(images))
    ]

    results = ray.get(tasks)

    # Save deformed images in parallel
    output_folder = 'outframes/'
    os.makedirs(output_folder, exist_ok=True)

    save_tasks = [
        save_image.remote(results[i], os.path.join(output_folder, frame_files[i]))
        for i in range(len(results))
    ]

    # Wait for all save tasks to complete
    ray.get(save_tasks)

    end = time.time()
    print(i , round(end - start, 2))
    calc_time.append((i, round(end - start, 2)))

    ray.shutdown()

    print("10 sec wait")
    time.sleep(10)


with open("timings.txt", "a") as f:
    f.write(f"inteli5,{no_workers},0,{calc_time}\n")

