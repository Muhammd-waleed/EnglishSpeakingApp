import pyaudio,wave
import numpy as np
import whisper_server

# Configuration settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Initialize PyAudio
p = pyaudio.PyAudio()

# # Open a stream to record audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")
frames = []

# # Record audio in chunks for the specified duration
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Finished recording.")

# Stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()


# Convert the frames to a numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

# # Load the Whisper model
model = whisper_server.load_model("base")


# # Transcribe the in-memory audio data
result = model.transcribe(data=audio_data, fp16=False)

# # Print the transcribed text
print("Transcribed Text:", result['text'])
