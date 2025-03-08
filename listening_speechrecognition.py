import pyaudio
import numpy as np
import noisereduce as nr
from queue import Queue
import threading
import time
import speech_recognition as sr
import ollama
from gtts import gTTS
import pygame
import io
import logging
from pathlib import Path
import wave
import os
import signal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self):
        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 0.005  # Adjust based on your environment
        self.SILENCE_DURATION = 2.5  # Reduced silence duration for better responsiveness
        self.MAX_RECORDING_SECONDS = 30  # Maximum recording time
        
        # State management
        self.audio_queue = Queue()
        self.response_queue = Queue()
        self.running = False
        self.listening_active = True  # Flag to toggle listening
        self.threads = []
        
        # Path for temporary files
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.init_audio()
        self.init_speech_recognition()
        self.init_audio_playback()
        
    def init_audio(self):
        """Initialize PyAudio"""
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            logger.info("Audio stream initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            raise
    
    def init_speech_recognition(self):
        """Initialize SpeechRecognition"""
        self.recognizer = sr.Recognizer()
        # Adjust parameters for better recognition
        self.recognizer.energy_threshold = 150
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.0
        
    def init_audio_playback(self):
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.init(frequency=24000)
            logger.info("Audio playback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio playback: {e}")
            raise
            
    def audio_input_thread(self):
        """Continuously capture audio input with enhanced silence detection"""
        silence_start = None
        recording_start = None
        
        while self.running:
            try:
                if not self.listening_active:
                    time.sleep(0.1)
                    continue
                    
                frames = []
                audio_detected = False
                recording_start = None
                
                # Listen for initial audio above threshold
                while self.running and self.listening_active:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Check if audio exceeds silence threshold
                    amplitude = np.max(np.abs(audio_data))
                    if amplitude > self.SILENCE_THRESHOLD:
                        if not audio_detected:
                            logger.info("Speech detected")
                            recording_start = time.time()
                        audio_detected = True
                        frames.append(audio_data)
                        silence_start = None
                        break
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
                # Continue recording until silence threshold is met or max time reached
                while self.running and self.listening_active and audio_detected:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    amplitude = np.max(np.abs(audio_data))
                    if amplitude > self.SILENCE_THRESHOLD:
                        frames.append(audio_data)
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.SILENCE_DURATION:
                            logger.info("Silence detected, processing audio")
                            break
                        frames.append(audio_data)  # Still append during short silences
                    
                    # Check if we've been recording too long
                    if recording_start and time.time() - recording_start > self.MAX_RECORDING_SECONDS:
                        logger.warning(f"Maximum recording time of {self.MAX_RECORDING_SECONDS}s reached")
                        break
                
                if frames and audio_detected:
                    # Process complete audio segment
                    audio_array = np.concatenate(frames)
                    self.audio_queue.put(audio_array)
                    
                    # Briefly pause listening while processing
                    self.listening_active = False
                    time.sleep(0.5)  # Give a moment for processing to start
            
            except Exception as e:
                logger.error(f"Error in audio input: {e}")
                time.sleep(1)
    
    def save_audio_to_wav(self, audio_data, filename=None):
        """Save numpy audio array to WAV file for SpeechRecognition"""
        if filename is None:
            filename = self.temp_dir / f"temp_{time.time()}.wav"
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.RATE)
            wf.writeframes(audio_int16.tobytes())
        return filename
    
    def process_audio_thread(self):
        """Process audio with enhanced error handling and fallbacks"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Noise reduction with fallback
                    try:
                        clean_audio = nr.reduce_noise(y=audio_data, sr=self.RATE)
                        logger.info("Noise reduction applied")
                    except Exception as e:
                        logger.error(f"Noise reduction failed: {e}")
                        clean_audio = audio_data
                    
                    # Speech-to-Text with multiple providers and fallbacks
                    text = self.transcribe_audio(clean_audio)
                    
                    if not text:
                        logger.warning("No text detected from audio")
                        self.listening_active = True
                        self.audio_queue.task_done()
                        continue
                    
                    logger.info(f"Transcribed: \"{text}\"")
                    
                    # Process with Ollama - directly use without verification
                    response_text = self.get_llm_response(text)
                    
                    if response_text:
                        # Text-to-Speech with fallback options
                        self.generate_speech(response_text)
                    
                    self.audio_queue.task_done()
                    self.listening_active = True
                else:
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Processing thread error: {e}")
                self.listening_active = True
                time.sleep(1)
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio with fallback options"""
        # Try Google Speech Recognition first
        try:
            temp_file = self.save_audio_to_wav(audio_data)
            
            with sr.AudioFile(str(temp_file)) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio)
            
            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)
            return text
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
        except Exception as e:
            logger.error(f"Google STT failed: {e}")
        
        
        return None
    
    def get_llm_response(self, text, retries=2):
        """Get response from Ollama with retry mechanism"""
        attempt = 0
        while attempt <= retries:
            try:
                logger.info(f"Querying Ollama (attempt {attempt+1}/{retries+1})")
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": text}],
                    options={"temperature": 0.7}  # Add some variability to responses
                )
                response_text = response["message"]["content"]
                logger.info(f"LLM response: {response_text[:50]}...")
                return response_text
            except Exception as e:
                logger.error(f"Ollama processing failed (attempt {attempt+1}): {e}")
                attempt += 1
                time.sleep(1)  # Wait before retry
        
        return "Sorry, I'm having trouble processing that request right now."
    
    def generate_speech(self, text):
        """Generate speech with enhanced error handling"""
        try:
            # Break long responses into sentences for more natural pauses
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            for sentence in sentences:
                if not sentence:
                    continue
                    
                tts = gTTS(sentence + '.', lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                self.response_queue.put(audio_buffer)
                
                # Wait for this sentence to be processed before adding the next
                while not self.response_queue.empty() and self.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            
            # Fallback response if TTS fails
            error_message = "I'm sorry, but I can't speak my response right now."
            logger.info(f"Fallback message: {error_message}")
    
    def playback_thread(self):
        """Play audio responses with improved handling"""
        while self.running:
            try:
                if not self.response_queue.empty():
                    audio_buffer = self.response_queue.get()
                    
                    # Stop any currently playing audio
                    pygame.mixer.music.stop()
                    
                    # Load and play new audio
                    pygame.mixer.music.load(audio_buffer)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy() and self.running:
                        time.sleep(0.1)
                    
                    audio_buffer.close()
                    self.response_queue.task_done()
                else:
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Playback error: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Close audio streams
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
        
        # Stop playback
        pygame.mixer.quit()
        
        # Remove temp files
        for file in self.temp_dir.glob("*.wav"):
            try:
                file.unlink()
            except:
                pass
        
        try:
            self.temp_dir.rmdir()
        except:
            pass
        
        logger.info("Cleanup complete")
    
    def start(self):
        """Start the voice assistant"""
        self.running = True
        
        # Start threads
        self.threads = [
            threading.Thread(target=self.audio_input_thread, daemon=True),
            threading.Thread(target=self.process_audio_thread, daemon=True),
            threading.Thread(target=self.playback_thread, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        logger.info("Voice assistant started. Speak to interact. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop the voice assistant"""
        logger.info("Shutting down...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.cleanup()

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Interrupt received, shutting down...")
    if 'assistant' in globals():
        assistant.stop()
    sys.exit(0)

if __name__ == "__main__":
    import sys
    
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        assistant = VoiceAssistant()
        assistant.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if 'assistant' in globals():
            assistant.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if 'assistant' in globals():
            assistant.stop()
        sys.exit(1)