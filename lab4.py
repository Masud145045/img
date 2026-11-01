import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)

# Apply Laplacian Filter
result = cv2.Laplacian(img, cv2.CV_64F)
result = cv2.convertScaleAbs(result)

# Apply Threshold
_, edge = cv2.threshold(result, 30, 255,
                        cv2.THRESH_BINARY)

# Display Images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(result, cmap="gray")
plt.title("Laplacian Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()