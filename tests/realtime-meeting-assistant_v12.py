from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import whisper
import threading
import numpy as np
import requests
import logging
import noisereduce as nr
from datetime import datetime
import sounddevice as sd
from typing import Dict, Any, Mapping, List, Optional, Union, Tuple
from numpy.typing import NDArray
import time
from collections import deque
import torch
from scipy.signal import butter, lfilter
import os
from scipy import signal

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
CHUNK_DURATION = 5.0
OVERLAP_DURATION = 2.0
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = int(RATE * CHUNK_DURATION)
OVERLAP_SIZE = int(RATE * OVERLAP_DURATION)
SILENCE_THRESHOLD = 0.0015
MIN_AUDIO_LENGTH = 0.05

# Ollama API Configuration (if you're using it)
OLLAMA_MACHINE_IP = "192.168.1.27"
OLLAMA_URL = f"http://{OLLAMA_MACHINE_IP}:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

class AudioPreprocessor:
    def __init__(self, order: int, lowcut: float, highcut: float, rate: int):
        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut
        self.rate = rate

    def butter_bandpass(self) -> tuple[np.ndarray, np.ndarray]:
        """Design a bandpass filter."""
        nyquist = 0.5 * self.rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def get_bandpass_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Butterworth bandpass filter coefficients for speech frequencies."""
        # Filter parameters
        nyquist = RATE / 2
        low_freq = 300  # Hz - typical lower bound for speech
        high_freq = 3400  # Hz - typical upper bound for speech
        order = 4  # Filter order
        
        # Normalize frequencies to Nyquist frequency
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Generate Butterworth filter coefficients
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data: NDArray) -> NDArray:
        b, a = self.get_bandpass_coefficients()
        filtered_data = lfilter(b, a, data)
        # Ensure we return just the filtered signal if lfilter returns a tuple
        return filtered_data[0] if isinstance(filtered_data, tuple) else filtered_data
        
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to improve audio quality."""
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Apply bandpass filter
        audio = self.apply_bandpass_filter(audio)
        
        # Apply noise reduction
        audio = nr.reduce_noise(
            y=audio,
            sr=RATE,
            prop_decrease=0.95,
            n_std_thresh_stationary=1.5,
            stationary=True
        )
        
        return audio
    
    def detect_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio chunk contains speech using energy-based VAD."""
        # Placeholder implementation of speech detection
        energy = np.sum(audio ** 2) / len(audio)
        return bool(energy > SILENCE_THRESHOLD)

class TranscriptionBuffer:
    def __init__(self, max_size: int = 5):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, text: str):
        with self.lock:
            self.buffer.append(text)
    
    def get_context(self) -> str:
        with self.lock:
            return " ".join(list(self.buffer))

class MeetingAnalyzer:
    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model_name = MODEL_NAME

    def analyze_text(self, text: str) -> str:
        """Analyze transcribed text for key points and action items."""
        try:
            prompt = f"""Analyze this meeting transcript and extract:
            1. Key points
            2. Action items
            3. Questions raised
            
            Transcript: {text}"""
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No analysis available')
            else:
                return f"Analysis failed: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return "Analysis unavailable"

class BatchTranscriber:
    def __init__(self):
        self.stream = None
        self.is_recording = False
        self.analyzer = MeetingAnalyzer()
        self.buffer = np.array([], dtype=np.float32)
        self.processing_thread = None
        self.buffer_lock = threading.Lock()
        self.preprocessor = AudioPreprocessor(order=5, lowcut=300.0, highcut=3000.0, rate=RATE)
        self.transcription_buffer = TranscriptionBuffer()
        
        # Load Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.whisper_model = whisper.load_model("medium").to(self.device)
        
    def start_recording(self):
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=int(RATE * 0.1)
        )
        self.stream.start()
        self.is_recording = True
        self.processing_thread = threading.Thread(target=self.process_chunks, daemon=True)
        self.processing_thread.start()
        logger.info("Audio recording and processing started")

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        logger.info("Audio recording stopped")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        if indata is not None:
            with self.buffer_lock:
                self.buffer = np.append(self.buffer, indata.flatten())

    def process_chunks(self):
        while self.is_recording or len(self.buffer) >= CHUNK_SIZE:
            if len(self.buffer) >= CHUNK_SIZE:
                with self.buffer_lock:
                    # Ensure consistent float32 type and shape
                    chunk = np.asarray(self.buffer[:CHUNK_SIZE], dtype=np.float32)
                    self.buffer = self.buffer[CHUNK_SIZE - OVERLAP_SIZE:]
                
                try:
                    # Normalize audio before preprocessing
                    if chunk.max() > 1.0:
                        chunk = chunk / 32768.0

                    processed_chunk = self.preprocessor.preprocess_audio(chunk)
                    
                    if self.preprocessor.detect_speech(processed_chunk):
                        context = self.transcription_buffer.get_context()
                        
                        # Convert to tensor for Whisper model
                        audio_tensor = torch.from_numpy(processed_chunk).float().to(self.device)
                        
                        result = self.whisper_model.transcribe(
                            audio_tensor,
                            language="en",
                            initial_prompt=context,
                            task="transcribe",
                            temperature=0.0,
                            no_speech_threshold=0.6,
                            logprob_threshold=-1.0
                        )
                        
                        text = "".join(result.get("text", [])).strip() if isinstance(result.get("text", []), list) else str(result.get("text", "")).strip()
                        if text:
                            text = self._clean_transcription(text)
                            self.transcription_buffer.add(text)
                            
                            analysis = self.analyzer.analyze_text(text)
                            socketio.emit('transcription', {
                                'text': text,
                                'analysis': analysis,
                                'timestamp': datetime.now().strftime("%H:%M:%S")
                            })
                            logger.debug(f"Processed chunk with text: {text}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    logger.exception("Full traceback:")

            time.sleep(0.1)

    def _clean_transcription(self, text: str) -> str:
        """Clean up transcription text with improved string handling."""
        fillers = ["um", "uh", "like", "you know", "sort of", "kind of", "basically"]
        cleaned_text = text.lower()
        
        # Remove filler words
        for filler in fillers:
            cleaned_text = cleaned_text.replace(f" {filler} ", " ")
        
        # Fix punctuation spacing
        for punct in [",", ".", "?", "!"]:
            cleaned_text = cleaned_text.replace(f" {punct}", punct)
        
        # Capitalize sentences
        sentences = [s.strip().capitalize() for s in cleaned_text.split(".") if s.strip()]
        cleaned_text = ". ".join(sentences)
        
        return cleaned_text.strip()

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process a single audio chunk with explicit type handling."""
        try:
            # Ensure float32 type and correct normalization
            audio_data = np.asarray(audio_chunk, dtype=np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Convert to tensor for Whisper model
            audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
            
            result = self.whisper_model.transcribe(
                audio_tensor,
                language='en',
                task='transcribe'
            )
            
            text = "".join(result.get("text", [])).strip() if isinstance(result.get("text", []), list) else str(result.get("text", "")).strip()
            return text
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            logger.exception("Full traceback:")
            return ""

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def handle_recording_start():
    global current_transcriber
    logger.info("Received start recording request")
    try:
        if current_transcriber:
            current_transcriber.stop_recording()
        current_transcriber = BatchTranscriber()
        current_transcriber.start_recording()
        logger.info("Recording started successfully")
        socketio.emit('status', {'message': 'Recording started'})
        return jsonify({"status": "success", "message": "Recording started"})
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        logger.exception("Full traceback:")
        return jsonify({"status": "error", "message": str(e)}), 500

@socketio.on('stop_recording')
def handle_recording_stop():
    global current_transcriber
    logger.info("Received stop recording request")
    try:
        if current_transcriber:
            current_transcriber.stop_recording()
            current_transcriber = None
        logger.info("Recording stopped successfully")
        socketio.emit('status', {'message': 'Recording stopped'})
        return jsonify({"status": "success", "message": "Recording stopped"})
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        logger.exception("Full traceback:")
        return jsonify({"status": "error", "message": str(e)}), 500

# Global variable for the current transcriber instance
current_transcriber = None

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)