import numpy as np
import librosa
from flask import request,jsonify
from voice_to_text import app,model

@app.route('/transcribe', methods=['POST'])
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