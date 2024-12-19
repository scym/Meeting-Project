import requests
import logging
import config

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.model_name = config.MODEL_NAME

    def analyze_text(self, text: str) -> str:
        try:
            prompt = f"""Analyze the following text:
            - Key points
            - Action items
            - Questions raised
            
            Transcript: {text}"""

            response = requests.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": prompt},
            )
            if response.status_code == 200:
                return response.json().get("response", "No analysis available")
            return f"Error: {response.status_code}"
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return "Analysis failed"
