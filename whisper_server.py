# whisper_server.py

from flask import Flask, request, jsonify
import whisper
import numpy as np
import librosa
import warnings

warnings.filterwarnings(action='ignore')

app = Flask(__name__)

# Load the Whisper model once when the server starts
model = whisper.load_model("base")

@app.route('/home')
def home_page():
    return 'This is Home Page'

@app.route('/transcribe', methods=['POST','GET'])
def transcribe_audio():
    # Get the uploaded audio data
    audio_bytes = request.data
    
    # Convert the audio data to a NumPy array
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Resample the audio data to 16000 Hz (required by Whisper)
    audio_data_resampled = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)
    
    # Perform the transcription
    result = model.transcribe(audio=audio_data_resampled)
    
    # Return the transcribed text as JSON
    return jsonify({"transcription": result['text']})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)
