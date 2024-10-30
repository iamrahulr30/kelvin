import numpy as np
import os, cv2, time
from tqdm import tqdm

calc_time = []

# Define input/output folders and load frame files
frames_folder = 'kaggle/frames_main/'
output_folder = 'outit/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_files = sorted(os.listdir(frames_folder))[0:100]
height, width, _ = cv2.imread(os.path.join(frames_folder, frame_files[0])).shape

# Force parameters
initial_power = 3000
max_power = 5000
final_power = 0
num_frames = len(frame_files)
decrease_phase_frames = int(num_frames * 0.1)  # Last 10% frames for decreasing phase
increase_phase_frames = num_frames - decrease_phase_frames

initial_y_position = 260.0
y_position_increment = 2
max_y_position = height - 1  # Cap to the image height

# Kelvinlet displacement function
def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + 10  # epsilon for stability
    u = (1 / (4 * np.pi)) * ((3 * -2 / r_norm) - (-2 / (r_norm ** 3))) * F
    return u

# Power calculation function with smooth increase and decrease
def calculate_power(frame_index):
    if frame_index < increase_phase_frames:
        # Increase phase
        return initial_power + (max_power - initial_power) * (frame_index / increase_phase_frames)
    else:
        # Decrease phase
        return max_power * (1 - ((frame_index - increase_phase_frames) / decrease_phase_frames))

def deform_frame(image, frame_index, width, height, max_y_position):
    # Calculate dynamic power and y-position for current frame
    force_power = calculate_power(frame_index)
    force_y_position = min(initial_y_position + (frame_index * y_position_increment), max_y_position)
    forces = [{'power': force_power, 'position': np.array([250.0, force_y_position])}]
    
    deformed_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            displacement = np.zeros(2)
            for force in forces:
                r = np.array([x, y]) - force['position']
                u = kelvinlet_displacement(r, force['power'])
                displacement += u
            new_x, new_y = np.clip(np.array([x, y]) + displacement, [0, 0], [width - 1, height - 1]).astype(int)
            deformed_image[y, x] = image[new_y, new_x]
    
    return deformed_image

# Process each frame sequentially
start_time = time.time()
for i, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
    if i % 10 == 0:
        calc_time.append([i , time.time() - start_time])
    image = cv2.imread(os.path.join(frames_folder, frame_file))
    deformed_frame = deform_frame(image, i, width, height, max_y_position)
    
    # Save the deformed frame
    output_path = os.path.join(output_folder, f"frame_{i + 30:04d}.png")
    cv2.imwrite(output_path, deformed_frame)
    print(f"Saved {output_path}")

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

print(calc_time)
