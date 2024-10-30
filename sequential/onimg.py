import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../data/imgs/tig.jpg') #out des change
height, width, _ = image.shape

forces = [
    {'power': 2700, 'position': np.array([260.0, 260.0])},
    {'power': 1300, 'position': np.array([width // 3, height // 3])},
    {'power': 2000, 'position': np.array([2 * width // 3, 2 * height // 3])}
]

# forces = [#mike
#     {'power': 1000, 'position': np.array([373.0 , 374.0])}, #ok
#     {'power': 5000, 'position': np.array([427.0, 679.0])}, 
#     {'power': 10000, 'position': np.array([34.0, 277.0])}
# ]

mu = 1.0
alpha = -2
epsilon = 10

def kelvinlet_displacement(r, F):
    r_norm = np.linalg.norm(r) + epsilon
    return (1 / (4 * np.pi * mu)) * ((3 * alpha / r_norm) - (alpha / (r_norm ** 3))) * F

deformed_image = np.zeros_like(image)
for y in range(height):
    for x in range(width):
        displacement = np.zeros(2)
        for force in forces:
            r = np.array([x, y]) - force['position']
            displacement += kelvinlet_displacement(r, force['power'])
        new_x, new_y = np.clip(np.array([x, y]) + displacement, [0, 0], [width - 1, height - 1]).astype(int)
        deformed_image[y, x] = image[new_y, new_x]

cv2.imwrite('../output/tig_deformed.jpg', deformed_image)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB))
plt.title('Deformed Image')

plt.show()
