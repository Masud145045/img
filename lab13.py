import pyttsx3
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from matplotlib.widgets import Button

engine = pyttsx3.init()
text = "Hi Masud. How are you? Let's take a cup of tea."

engine.save_to_file(text, "output.wav")
engine.runAndWait()
print("Speech saved as output.wav")

fs, speech = wavfile.read("output.wav")

if len(speech.shape) > 1:
    speech = speech.mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

ax.plot(speech)
ax.set_title("Text-to-Speech Signal")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.grid()

def play_speech(event):
    sd.stop()
    sd.play(speech, fs)

def stop_speech(event):
    sd.stop()

ax_play = plt.axes([0.30, 0.07, 0.20, 0.08])
ax_stop = plt.axes([0.55, 0.07, 0.20, 0.08])

button_play = Button(ax_play, "Play Speech")
button_stop = Button(ax_stop, "Stop")

button_play.on_clicked(play_speech)
button_stop.on_clicked(stop_speech)

plt.show()