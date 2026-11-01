import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)

# Smoothing Filters
mean = cv2.blur(img, (3, 3))
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 3)

# Sharpening Filter
laplacian = cv2.Laplacian(img, cv2.CV_32F)
sharp = cv2.convertScaleAbs(img.astype("float32") - laplacian)

# Display Results
images = [img, mean, gaussian, median, sharp]
titles = ["Original", "Mean Filter", "Gaussian Filter", "Median Filter", "Laplacian Sharpening"]

plt.figure(figsize=(12, 7))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()