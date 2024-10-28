from flask import Flask
import whisper

app=Flask()

# Load the Whisper model once when the server starts
model=whisper.load_model("base")

import routes

