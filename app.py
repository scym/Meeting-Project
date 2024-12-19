from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from utils.transcription import BatchTranscriber
from utils.logging_config import configure_logging
import config

# Configure logging
logger = configure_logging()

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

current_transcriber = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def handle_recording_start():
    global current_transcriber
    logger.info("Start recording requested")
    try:
        if current_transcriber:
            logger.info("Stopping existing transcriber")
            current_transcriber.stop_recording()
        current_transcriber = BatchTranscriber(socketio)
        current_transcriber.start_recording()
        logger.info("Recording started successfully")
        socketio.emit('status', {'message': 'Recording started'})
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@socketio.on('stop_recording')
def handle_recording_stop():
    global current_transcriber
    logger.info("Stop recording requested")
    try:
        if current_transcriber:
            logger.info("Stopping transcriber")
            current_transcriber.stop_recording()
            current_transcriber = None
        logger.info("Recording stopped successfully")
        socketio.emit('status', {'message': 'Recording stopped'})
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
