
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("character.jpg", 0)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
chars = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if cv2.contourArea(c) > 100:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)
        chars.append(img[y:y+h, x:x+w])

plt.figure(figsize=(12, 7))
plt.subplot(3, 5, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 5, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Green Box")
plt.axis("off")

for i, c in enumerate(chars):
    plt.subplot(3, 5, i+3)
    plt.imshow(c, cmap="gray")
    plt.title("Character " + str(i+1))
    plt.axis("off")

plt.tight_layout()
plt.show()
