import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("speech.wav")

frame = int(0.025 * sr)
energy = []
zcr = []

for i in range(0, len(y), frame):
    x = y[i:i+frame]

    if len(x) > 0:
        energy.append(np.mean(x**2))
        zcr.append(np.mean(np.abs(np.diff(np.sign(x)))) / 2)

energy = np.array(energy)
zcr = np.array(zcr)

for i in range(len(energy)):
    if energy[i] < 0.001:
        print("Silence")
    elif zcr[i] > 0.15:
        print("Unvoiced")
    else:
        print("Voiced")

plt.plot(y)
plt.title("Speech Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()