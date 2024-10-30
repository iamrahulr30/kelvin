import cv2
import numpy as np
import matplotlib.pyplot as plt

deformed_image = cv2.imread('deformed_image_parallel.jpg')
height, width, channels = deformed_image.shape

forces = [
    {'power': 500, 'position': np.array([width // 2, height // 2])},
    {'power': 300, 'position': np.array([width // 3, height // 3])},
    {'power': 200, 'position': np.array([2 * width // 3, 2 * height // 3])}
]

mu = 1.0
alpha = 1.0
epsilon = 1e-6

def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    u = (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F
    return u

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
        original_coords = np.linalg.solve(A, b)

        orig_x, orig_y = np.clip(original_coords, [0, 0], [width - 1, height - 1]).astype(int)
        recovered_image[y, x] = deformed_image[orig_y, orig_x]

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB))
plt.title('Deformed Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB))
plt.title('Recovered Image (Sequential)')

plt.show()
