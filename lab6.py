import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)

# Sobel
sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

# Prewitt
kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
px = cv2.filter2D(img, cv2.CV_64F, kx)
py = cv2.filter2D(img, cv2.CV_64F, ky)
prewitt = cv2.convertScaleAbs(cv2.magnitude(px, py))

# Canny
canny = cv2.Canny(img, 100, 200)

# Roberts
rx = cv2.filter2D(img, cv2.CV_64F, np.array([[1,0],[0,-1]]))
ry = cv2.filter2D(img, cv2.CV_64F, np.array([[0,1],[-1,0]]))
roberts = cv2.convertScaleAbs(cv2.magnitude(rx, ry))

# Display Results
images = [img, sobel, prewitt, canny, roberts]
titles = ["Original", "Sobel", "Prewitt", "Canny", "Roberts"]

plt.figure(figsize=(12, 7))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()