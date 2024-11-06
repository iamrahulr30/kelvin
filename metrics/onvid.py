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

    images = [
        cv2.resize(cv2.imread(os.path.join(frames_folder, file)), (512, 512)) for file in frame_files
    ]

    total_iterations = end_frame - start_frame
    midpoint = total_iterations // 2

    initial_y = 260
    final_y = 250
    initial_power = 3000
    max_power = 4000

    mu = 1.0
    alpha = -2
    epsilon = 1e-5

    def kelvinlet_displacement(r, F):
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True) + epsilon
        u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
        return np.nan_to_num(u)

    @ray.remote
    def process_frame(image, iter_index, power, y_position):
        height, width, _ = image.shape
        
        x_position = 260.0

        forces = [{'power': power, 'position': np.array([x_position, y_position])}]

        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        displacement = np.zeros((height, width, 2))

        for force in forces:
            r = np.stack([x_coords - force['position'][0], y_coords - force['position'][1]], axis=-1)
            u = kelvinlet_displacement(r, force['power'])
            displacement += u

        new_x = np.clip((x_coords + displacement[..., 0]).astype(np.float32), 0, width - 1)
        new_y = np.clip((y_coords + displacement[..., 1]).astype(np.float32), 0, height - 1)

        new_x = np.clip(new_x.astype(int), 0, width - 1)
        new_y = np.clip(new_y.astype(int), 0, height - 1)

        deformed_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                deformed_image[i, j] = image[new_y[i, j], new_x[i, j]]

        return deformed_image

    @ray.remote
    def save_image(image, filename):
        cv2.imwrite(filename, image)

    power_values = np.zeros(total_iterations)
    power_values[:midpoint] = np.linspace(initial_power, max_power, midpoint)
    power_values[midpoint:] = np.linspace(max_power, 0, total_iterations - midpoint)

    y_positions = np.linspace(initial_y, final_y, total_iterations)

    tasks = [
        process_frame.remote(images[i], start_frame + i, power_values[i], y_positions[i])
        for i in range(len(images))
    ]

    results = ray.get(tasks)

    output_folder = 'outframes/'
    os.makedirs(output_folder, exist_ok=True)

    save_tasks = [
        save_image.remote(results[i], os.path.join(output_folder, frame_files[i]))
        for i in range(len(results))
    ]

    ray.get(save_tasks)

    end = time.time()
    print(i , round(end - start, 2))
    calc_time.append((i, round(end - start, 2)))

    ray.shutdown()

    print("10 sec wait")
    time.sleep(10)

with open("timings.txt", "a") as f:
    f.write(f"inteli5,{no_workers},0,{calc_time}\n")
