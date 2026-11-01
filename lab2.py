
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread("image.png", 0)

# Binary Image
_, binary = cv2.threshold(img, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find Rice
contours, _ = cv2.findContours(binary,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
selected = np.zeros_like(img)
count = 0

# Rice Measurement
for c in contours:
    area = cv2.contourArea(c)

    if 3000 <= area <= 25000:
        count += 1
        x, y, w, h = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        major = max(w, h)

        print("Rice:", count, "Area:", area)
        print("Major Axis:", major)
        print("Perimeter:", round(perimeter, 2))

        cv2.rectangle(output, (x,y), (x+w,y+h),
                      (0,255,0), 2)
        selected[y:y+h, x:x+w] = binary[y:y+h, x:x+w]

print("Total Rice:", count)

# Display Output
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Rice")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(selected, cmap="gray")
plt.title("Selected Rice")
plt.axis("off")

plt.tight_layout()
plt.show()
