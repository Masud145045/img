import pyttsx3

text = input("Enter text: ")

engine = pyttsx3.init()
engine.say(text)
engine.save_to_file(text, "record_audio.wav")
engine.runAndWait()

print("Speech played and saved successfully!")