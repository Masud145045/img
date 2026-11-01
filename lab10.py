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

energy = []
zcr = []
times = []

for i in range(0, len(audio) - frame + 1, step):
    data = audio[i:i+frame]
    energy.append(np.mean(data**2))
    zcr.append(np.sum(data[:-1] * data[1:] < 0) / len(data))
    times.append(i / rate)

energy = np.array(energy)
zcr = np.array(zcr)

limit = max(0.0001, 0.02 * max(energy))
zlimit = np.median(zcr[energy >= limit])

regions = ["Silence" if e < limit else "Voiced" if z <= zlimit else "Unvoiced"
           for e, z in zip(energy, zcr)]

for name in ["Voiced", "Unvoiced", "Silence"]:
    print(name, ":", round(regions.count(name) * step / rate, 2), "seconds")

plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(audio)) / rate, audio)

for t, r in zip(times, regions):
    color = "green" if r == "Voiced" else "orange" if r == "Unvoiced" else "lightgray"
    plt.axvspan(t, t + step/rate, color=color, alpha=0.3)

plt.title("Speech Region Classification")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()