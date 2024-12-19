import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr
from typing import Tuple
from numpy.typing import NDArray
from scipy import signal
import config

class AudioProcessing:
    def __init__(self, rate, lowcut, highcut, order):
        self.rate = rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def butter_bandpass(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = 0.5 * self.rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype="band")
        return b, a

    def process(self, data):
        b, a = self.butter_bandpass()
        lfilter(b, a, data)

class AudioPreprocessor:
    def __init__(self, order, lowcut, highcut, rate):
        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut
        self.rate = rate

    def butter_bandpass(self):
        nyquist = 0.5 * self.rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        return butter(self.order, [low, high], btype="band")

    def apply_bandpass_filter(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        b, a = self.butter_bandpass()
        filtered_data = lfilter(b, a, data)
        return np.asarray(filtered_data)  # Ensure this is a single ndarray

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        return nr.reduce_noise(y=audio, sr=self.rate, prop_decrease=0.95)

    def detect_speech(self, audio: np.ndarray) -> bool:
        return bool(np.sum(audio**2) / len(audio) > 0.0015)
