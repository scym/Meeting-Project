# Configuration variables

CHUNK_DURATION = 5.0
OVERLAP_DURATION = 2.0
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = int(RATE * CHUNK_DURATION)
OVERLAP_SIZE = int(RATE * OVERLAP_DURATION)
SILENCE_THRESHOLD = 0.0015
MIN_AUDIO_LENGTH = 0.05

OLLAMA_MACHINE_IP = "192.168.1.27"
OLLAMA_URL = f"http://{OLLAMA_MACHINE_IP}:11434/api/generate"
MODEL_NAME = "llama3.2:latest"
