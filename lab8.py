import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("coins.png", 0)

# Threshold
_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Power-Law Enhancement
power = np.uint8(255 * (img / 255.0) ** 0.5)

# High-Pass Filter
mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
high = cv2.filter2D(img, -1, mask)

# Display Images
images = [img, threshold, power, high]
titles = ["Original", "Threshold", "Power-Law", "High-Pass"]

plt.figure(figsize=(10, 7))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()