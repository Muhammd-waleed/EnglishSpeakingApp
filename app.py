from flask import Flask, request, render_template, jsonify
import whisper
import numpy as np

app = Flask(__name__)

# Load the Whisper model
model = whisper.load_model("base")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_data = request.data  # Get audio data from the request

    # Convert bytes to a NumPy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)  # Assuming 16-bit PCM format

    
    # Here you would process the audio data
    result = model.transcribe(audio=audio_array)  # Process the audio
    
    return jsonify({"transcription": result['text']})

if __name__ == '__main__':
    app.run(debug=True)
