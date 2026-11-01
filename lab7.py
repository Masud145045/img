import cv2
import matplotlib.pyplot as plt

img = cv2.imread("coins.png", 0)

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
count = 0

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        count += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
        cv2.putText(output, str(count), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("Coin", count, "Area:", round(area, 2))

print("Total Coins:", count)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Coins")
plt.axis("off")

plt.show()