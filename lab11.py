
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

rate, audio = wavfile.read("speech.wav")

if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = audio.astype(float)
audio = audio / (np.max(np.abs(audio)) + 1e-10)

frame = int(rate * 0.025)
step = int(rate * 0.010)
length = min(2 * rate, len(audio))

energy = [np.mean(audio[i:i+frame]**2)
          for i in range(0, len(audio)-frame+1, step)]

window = max(1, length // step)
total = np.convolve(energy, np.ones(window), "valid")

start = np.argmax(total) * step if len(total) else 0
signal = audio[start:start+length]
time = np.arange(len(signal)) / rate

def quantize(x, n):
    step = 2 / n
    return -1 + (np.clip(np.floor((x+1)/step), 0, n-1) + 0.5) * step

signals = [signal, quantize(signal, 4),
           quantize(signal, 8), quantize(signal, 16)]

titles = ["Original Signal", "4-Level", "8-Level", "16-Level"]
colors = ["blue", "orange", "green", "purple"]

plt.figure(figsize=(12, 9))

for i in range(4):
    plt.subplot(4, 1, i+1)
    if i == 0:
        plt.plot(time, signals[i], color=colors[i])
    else:
        plt.step(time, signals[i], where="post", color=colors[i])
    plt.title(titles[i], color=colors[i])
    plt.ylabel("Amplitude")
    plt.ylim(-1.05, 1.05)
    plt.grid()

plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

print("Sampling Rate:", rate, "Hz")
print("Segment Duration:", round(len(signal)/rate, 2), "seconds")
print("RMS Energy:", round(np.sqrt(np.mean(signal**2)), 4))
