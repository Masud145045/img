import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)

# 3x3 Average Mask
mask = np.ones((3, 3)) / 9

# Apply Convolution
result = cv2.filter2D(img, -1, mask)

# Show Images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Convolution")
plt.axis("off")

plt.show()