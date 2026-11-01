import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("speech.wav")
y = np.clip(y, -1, 1)

def quantize(y, L):
    q = np.round((y + 1) * (L - 1) / 2)
    return q * 2 / (L - 1) - 1

y4 = quantize(y, 4)
y8 = quantize(y, 8)
y16 = quantize(y, 16)

plt.figure(figsize=(10, 7))

plt.subplot(4, 1, 1)
plt.plot(y)
plt.title("Original Speech Signal")

plt.subplot(4, 1, 2)
plt.plot(y4)
plt.title("4-Level Quantization")

plt.subplot(4, 1, 3)
plt.plot(y8)
plt.title("8-Level Quantization")

plt.subplot(4, 1, 4)
plt.plot(y16)
plt.title("16-Level Quantization")

plt.tight_layout()
plt.show()