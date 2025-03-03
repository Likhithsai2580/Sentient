import pyaudio
import numpy as np
import noisereduce as nr
from queue import Queue
import threading
import time
import whisper
import ollama
from gtts import gTTS
import pygame
import io
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01  # Adjust based on your environment
SILENCE_DURATION = 2  # Seconds of silence to consider end of speech

# Initialize components
audio_queue = Queue()
response_queue = Queue()
running = True

# Load models
try:
    stt_model = whisper.load_model("tiny.en")  # Using tiny model for speed
    logger.info("Whisper STT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    exit(1)

# Initialize PyAudio
try:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    logger.info("Audio stream initialized")
except Exception as e:
    logger.error(f"Failed to initialize audio stream: {e}")
    exit(1)

# Initialize pygame for audio playback
pygame.mixer.init()

def audio_input_thread():
    """Continuously capture audio input"""
    global running
    silence_start = None
    
    while running:
        try:
            frames = []
            audio_detected = False
            
            # Collect audio until silence is detected
            while running:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Check if audio exceeds silence threshold
                amplitude = np.max(np.abs(audio_data))
                if amplitude > SILENCE_THRESHOLD:
                    if not audio_detected:
                        logger.info("Speech detected")
                    audio_detected = True
                    frames.append(audio_data)
                    silence_start = None
                elif audio_detected:
                    # Start counting silence duration
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        # Enough silence detected, process the audio
                        break
                
                # Prevent excessive memory usage
                if len(frames) > 1000:  # ~25 seconds at 40ms chunks
                    logger.warning("Audio input too long, processing partial input")
                    break
            
            if frames and audio_detected:
                audio_array = np.concatenate(frames)
                audio_queue.put(audio_array)
        
        except Exception as e:
            logger.error(f"Error in audio input: {e}")
            time.sleep(1)  # Prevent tight loop on error

def process_audio_thread():
    """Process audio, remove noise, convert to text, get response, and speak"""
    while running:
        try:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Noise reduction
                try:
                    clean_audio = nr.reduce_noise(y=audio_data, sr=RATE)
                    logger.info("Noise reduction applied")
                except Exception as e:
                    logger.error(f"Noise reduction failed: {e}")
                    clean_audio = audio_data  # Fallback to original audio
                
                # Speech-to-Text
                try:
                    result = stt_model.transcribe(clean_audio, fp16=False)
                    text = result["text"].strip()
                    if not text:
                        logger.warning("No text detected from audio")
                        continue
                    logger.info(f"Transcribed text: {text}")
                except Exception as e:
                    logger.error(f"STT failed: {e}")
                    continue
                
                # Process with Ollama Llama3.2:3b
                try:
                    response = ollama.chat(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": text}]
                    )
                    response_text = response["message"]["content"]
                    logger.info(f"LLM response: {response_text}")
                except Exception as e:
                    logger.error(f"Ollama processing failed: {e}")
                    response_text = "Sorry, I couldn't process that."
                
                # Text-to-Speech
                try:
                    tts = gTTS(response_text, lang='en')
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    response_queue.put(audio_buffer)
                except Exception as e:
                    logger.error(f"TTS failed: {e}")
                
                audio_queue.task_done()
            else:
                time.sleep(0.1)  # Prevent busy waiting
        
        except Exception as e:
            logger.error(f"Processing thread error: {e}")
            time.sleep(1)

def playback_thread():
    """Play audio responses"""
    while running:
        try:
            if not response_queue.empty():
                audio_buffer = response_queue.get()
                
                # Stop any currently playing audio
                pygame.mixer.music.stop()
                
                # Load and play new audio
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy() and running:
                    time.sleep(0.1)
                
                audio_buffer.close()
                response_queue.task_done()
            else:
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Playback error: {e}")
            time.sleep(1)

def main():
    global running
    
    # Start threads
    threads = [
        threading.Thread(target=audio_input_thread, daemon=True),
        threading.Thread(target=process_audio_thread, daemon=True),
        threading.Thread(target=playback_thread, daemon=True)
    ]
    
    for thread in threads:
        thread.start()
    
    logger.info("Voice assistant started. Speak to interact. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        running = False
    
    # Cleanup
    for thread in threads:
        thread.join()
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.mixer.quit()
    logger.info("Shutdown complete")

if __name__ == "__main__":
    # Verify Ollama is running locally
    try:
        ollama.list()
        logger.info("Ollama service detected")
    except Exception as e:
        logger.error(f"Ollama not running locally: {e}")
        logger.info("Please start Ollama with 'ollama serve' and pull llama3.2:3b")
        exit(1)
    
    main()
