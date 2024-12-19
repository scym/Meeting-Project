import numpy as np
import sounddevice as sd
import threading
import torch
import whisper
from datetime import datetime
from collections import deque
from scipy.signal import butter, lfilter
import noisereduce as nr
import logging
import time
from flask_socketio import SocketIO, emit

socketio = SocketIO()
from typing import Tuple, List
from config import RATE, CHANNELS, CHUNK_SIZE, OVERLAP_SIZE, SILENCE_THRESHOLD, OLLAMA_URL, MODEL_NAME

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, order: int, lowcut: float, highcut: float, rate: int):
        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut
        self.rate = rate

    def butter_bandpass(self) -> Tuple[np.ndarray, np.ndarray]:
        nyquist = 0.5 * self.rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def get_bandpass_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        nyquist = RATE / 2
        low_freq = 300  # Hz - typical lower bound for speech
        high_freq = 3400  # Hz - typical upper bound for speech
        order = 4  # Filter order
        
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        b, a = self.get_bandpass_coefficients()
        filtered_data = lfilter(b, a, data)
        return filtered_data[0] if isinstance(filtered_data, tuple) else filtered_data
        
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        audio = self.apply_bandpass_filter(audio)
        audio = nr.reduce_noise(
            y=audio,
            sr=RATE,
            prop_decrease=0.95,
            n_std_thresh_stationary=1.5,
            stationary=True
        )
        return audio
    
    def detect_speech(self, audio: np.ndarray) -> bool:
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
        try:
            # Placeholder for actual analysis logic
            return "Analysis result"
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return ""

class BatchTranscriber:
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
        self.stream = None
        self.is_recording = False
        self.analyzer = MeetingAnalyzer()
        self.buffer = np.array([], dtype=np.float32)
        self.processing_thread = None
        self.buffer_lock = threading.Lock()
        self.preprocessor = AudioPreprocessor(order=5, lowcut=300.0, highcut=3000.0, rate=RATE)
        self.transcription_buffer = TranscriptionBuffer()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.whisper_model = whisper.load_model("medium").to(self.device)
        logger.info("Whisper model loaded successfully")
        
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
            logger.debug(f"Audio callback received {frames} frames")

    def process_chunks(self):
        while self.is_recording or len(self.buffer) >= CHUNK_SIZE:
            if len(self.buffer) >= CHUNK_SIZE:
                with self.buffer_lock:
                    chunk = np.asarray(self.buffer[:CHUNK_SIZE], dtype=np.float32)
                    self.buffer = self.buffer[CHUNK_SIZE - OVERLAP_SIZE:]
                logger.debug(f"Processing chunk of size {len(chunk)}")
                
                try:
                    if chunk.max() > 1.0:
                        chunk = chunk / 32768.0

                    processed_chunk = self.preprocessor.preprocess_audio(chunk)
                    
                    if self.preprocessor.detect_speech(processed_chunk):
                        context = self.transcription_buffer.get_context()
                        
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
                            self.socketio.emit('transcription', {
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
        fillers = ["um", "uh", "like", "you know", "sort of", "kind of", "basically"]
        cleaned_text = text.lower()
        
        for filler in fillers:
            cleaned_text = cleaned_text.replace(filler, "")
        
        for punct in [",", ".", "?", "!"]:
            cleaned_text = cleaned_text.replace(f" {punct}", punct)
        
        sentences = [s.strip().capitalize() for s in cleaned_text.split(".") if s.strip()]
        cleaned_text = ". ".join(sentences)
        
        return cleaned_text.strip()

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> str:
        try:
            processed_chunk = self.preprocessor.preprocess_audio(audio_chunk)
            if self.preprocessor.detect_speech(processed_chunk):
                audio_tensor = torch.from_numpy(processed_chunk).float().to(self.device)
                result = self.whisper_model.transcribe(audio_tensor, language="en")
                text = result.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                return text.strip()
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
        return ""
