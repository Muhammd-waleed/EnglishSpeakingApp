# transcribe_client.py

import pyaudio
import requests

# Configuration settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Flask server URL
server_url = "http://127.0.0.1:5000/transcribe"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream to record audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

# Capture audio data from the microphone
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert the recorded frames to bytes
audio_data = b''.join(frames)

# Send the audio data to the Flask server
response = requests.post(server_url, data=audio_data)

# Check if the request was successful
if response.status_code == 200:
    # Print the transcription result
    print("Transcribed Text:", response.json().get("transcription"))
else:
    # Print the error message
    print("Error:", response.json().get("error"))
