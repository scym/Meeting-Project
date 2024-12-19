import requests
import json
import logging
import config

logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.model_name = config.MODEL_NAME

    def analyze_text(self, text: str) -> str:
        try:
            prompt = f"""Analyze this meeting transcript and extract:
            1. Key points
            2. Action items
            3. Questions raised
            
            Transcript: {text}"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            logger.debug(f"Sending request to {self.ollama_url} with payload: {payload}")
            response = requests.post(self.ollama_url, json=payload, timeout=10)

            logger.debug(f"Raw API Response: {response.text}")
            response.raise_for_status()  # Raise HTTP errors if any
            
            cleaned_response = response.text.strip()
            try:
                parsed_response = json.loads(cleaned_response)
                return parsed_response.get("response", "No analysis available")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Response content: {cleaned_response}")
                return "Analysis failed: Invalid JSON format"

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error during analysis: {e}")
            return "Analysis failed: HTTP error"
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return "Analysis unavailable"
