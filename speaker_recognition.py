import speech_recognition as sr
import numpy as np
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import os
import time
import threading
import queue
import logging

class SimpleSpeakerRecognition:
    def __init__(self, sample_rate=16000):
        """
        Simple speaker recognition system with basic embedding extraction
        """
        self.sample_rate = sample_rate
        self.speaker_embeddings = {}
    
    def extract_speaker_embedding(self, audio_path):
        """
        Extract basic audio features as a simple embedding
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            np.ndarray: Audio feature embedding
        """
        # Load audio file
        signal, sample_rate = sf.read(audio_path)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            # Basic resampling
            signal = np.interp(
                np.linspace(0, len(signal), int(len(signal) * self.sample_rate / sample_rate)),
                np.arange(len(signal)),
                signal
            )
        
        # Extract basic features
        # Use RMS energy and zero-crossing rate as simple features
        rms = np.sqrt(np.mean(signal**2))
        zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
        
        # Compute spectral centroid (basic spectral feature)
        spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(np.fft.rfft(signal))) / np.sum(np.abs(np.fft.rfft(signal)))
        
        # Combine features into a simple embedding
        embedding = np.array([rms, zcr, spectral_centroid])
        return embedding
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
        
        Returns:
            float: Similarity score
        """
        # Euclidean distance (lower is more similar)
        distance = np.linalg.norm(embedding1 - embedding2)
        # Convert to similarity score (0-1 range, where 1 is most similar)
        return 1 / (1 + distance)

class LiveSpeakerRecognitionSystem:
    def __init__(self, sample_rate=16000, chunk_duration=3):
        """
        Initialize live speaker recognition system
        
        Args:
            sample_rate (int): Audio sampling rate
            chunk_duration (int): Duration of audio chunks for processing
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Audio Parameters
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        
        # Speaker Recognition Setup
        try:
            # Attempt to use advanced model
            import speechbrain as sb
            from speechbrain.pretrained import SpeakerRecognition
            self.logger.info("Using advanced SpeechBrain speaker recognition")
            self.verification_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            self.recognition_type = "advanced"
        except Exception as e:
            self.logger.warning(f"Advanced model loading failed: {e}")
            self.logger.info("Falling back to simple speaker recognition")
            self.verification_model = SimpleSpeakerRecognition(sample_rate)
            self.recognition_type = "simple"
        
        # Speaker Enrollment Management
        self.speaker_embeddings = {}
        
        # Audio Capture Queue
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
    
    def save_audio_chunk(self, chunk, filename):
        """
        Save audio chunk to a temporary file
        
        Args:
            chunk (np.ndarray): Audio data
            filename (str): Output filename
        """
        sf.write(filename, chunk, self.sample_rate)
    
    def extract_speaker_embedding(self, audio_path):
        """
        Extract speaker embedding 
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            Embedding representation
        """
        if self.recognition_type == "advanced":
            # Use SpeechBrain method
            import torch
            signal, fs = torchaudio.load(audio_path)
            
            # Resample if needed
            if fs != self.sample_rate:
                resampler = torchaudio.transforms.Resample(fs, self.sample_rate)
                signal = resampler(signal)
            
            embedding = self.verification_model.encode_batch(signal)
            return embedding.squeeze()
        else:
            # Use simple recognition method
            return self.verification_model.extract_speaker_embedding(audio_path)
    
    def audio_capture_thread(self):
        """
        Continuously capture audio in a separate thread
        """
        with sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            dtype='float32'
        ) as stream:
            while not self.stop_event.is_set():
                chunk = np.zeros((self.sample_rate * self.chunk_duration,), dtype='float32')
                for i in range(int(self.chunk_duration)):
                    chunk_part, _ = stream.read(self.sample_rate)
                    chunk[i*self.sample_rate:(i+1)*self.sample_rate] = chunk_part.flatten()
                
                self.audio_queue.put(chunk)
    
    def enroll_speaker(self, name):
        """
        Enroll a new speaker
        
        Args:
            name (str): Speaker name
        """
        self.logger.info(f"Enrolling speaker {name}. Please speak for 3-5 seconds...")
        
        # Capture audio chunk
        chunk = self.audio_queue.get()
        temp_file = f"enrollment_{name}.wav"
        self.save_audio_chunk(chunk, temp_file)
        
        try:
            # Extract and store embedding
            embedding = self.extract_speaker_embedding(temp_file)
            self.speaker_embeddings[name] = embedding
            
            self.logger.info(f"Speaker {name} enrolled successfully!")
        except Exception as e:
            self.logger.error(f"Error enrolling speaker {name}: {e}")
        finally:
            # Always remove temporary file
            os.remove(temp_file)
    
    def recognize_speech(self, audio_chunk):
        """
        Perform speech recognition on audio chunk
        
        Args:
            audio_chunk (np.ndarray): Audio data
        
        Returns:
            str: Recognized text
        """
        temp_file = "current_speech.wav"
        self.save_audio_chunk(audio_chunk, temp_file)
        
        with sr.AudioFile(temp_file) as source:
            audio = self.recognizer.record(source)
        
        os.remove(temp_file)
        
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Speech was unintelligible"
        except sr.RequestError:
            return "Could not request results"
    
    def identify_speaker(self, audio_chunk):
        """
        Identify the speaker of an audio chunk
        
        Args:
            audio_chunk (np.ndarray): Audio data
        
        Returns:
            dict: Speaker identification results
        """
        temp_file = "current_speaker.wav"
        self.save_audio_chunk(audio_chunk, temp_file)
        
        try:
            test_embedding = self.extract_speaker_embedding(temp_file)
            
            # Compare against enrolled speakers
            similarities = {}
            for speaker_name, enrolled_embedding in self.speaker_embeddings.items():
                if self.recognition_type == "advanced":
                    # Use SpeechBrain similarity
                    score = self.verification_model.compute_similarity(
                        test_embedding.unsqueeze(0), 
                        enrolled_embedding.unsqueeze(0)
                    )
                else:
                    # Use simple recognition similarity
                    score = self.verification_model.compute_similarity(
                        test_embedding, 
                        enrolled_embedding
                    )
                similarities[speaker_name] = score.item() if hasattr(score, 'item') else score
            
            os.remove(temp_file)
            return similarities
        except Exception as e:
            self.logger.error(f"Speaker identification error: {e}")
            return {}
    
    def run_live_recognition(self):
        """
        Run live speaker recognition and speech detection
        """
        # Start audio capture thread
        capture_thread = threading.Thread(target=self.audio_capture_thread)
        capture_thread.start()
        
        try:
            while not self.stop_event.is_set():
                # Get audio chunk
                chunk = self.audio_queue.get()
                
                # Perform speech recognition
                speech_text = self.recognize_speech(chunk)
                
                # Identify speaker
                speaker_results = self.identify_speaker(chunk)
                
                # Print results
                print("\n--- Recognition Results ---")
                print("Speech Text:", speech_text)
                
                if speaker_results:
                    print("Speaker Similarities:")
                    for speaker, score in sorted(speaker_results.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {speaker}: {score:.4f}")
                
                # Optional: Add a small delay to prevent rapid processing
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("Stopping live recognition...")
        finally:
            self.stop_event.set()
            capture_thread.join()

def main():
    # Create recognition system
    recognition_system = LiveSpeakerRecognitionSystem()
    
    # Enroll speakers before starting recognition
    recognition_system.enroll_speaker("Alice")
    recognition_system.enroll_speaker("Bob")
    
    # Start live recognition
    recognition_system.run_live_recognition()

if __name__ == "__main__":
    main()