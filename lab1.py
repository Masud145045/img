
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("image.jpg", 0)

plt.figure(figsize=(10, 7))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Histogram
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("Histogram")

# Low Pass Filter
low = np.ones((3, 3)) / 9
low_img = cv2.filter2D(img, -1, low)

plt.subplot(2, 2, 3)
plt.imshow(low_img, cmap='gray')
plt.title("Low Pass Filter")
plt.axis("off")

# High Pass Filter
high = np.array([[-1, -1, -1],
                 [-1,  9, -1],
                 [-1, -1, -1]])

high_img = cv2.filter2D(img, -1, high)

plt.subplot(2, 2, 4)
plt.imshow(high_img, cmap='gray')
plt.title("High Pass Filter")
plt.axis("off")

plt.tight_layout(pad=3)
plt.show()
