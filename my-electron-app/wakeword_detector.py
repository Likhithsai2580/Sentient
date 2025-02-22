"""
This script implements a wake word detector using OpenWakeWord library for local wake word detection
and PyAudio for microphone input. Upon detecting the wake word "Hey Sentient", it launches an Electron
application and a Python server in separate subprocesses. It also includes functionality to gracefully
shutdown the application and server when the wake word detection is no longer needed or when the script is interrupted.
"""
import webrtcvad
from openwakeword.model import Model
import numpy as np
import pyaudio
import threading
from queue import Queue
import time
import os
import subprocess
from time import sleep

class WakeWordDetector:
    """
    A class to detect a specific wake word using OpenWakeWord models and trigger actions upon detection.
    It manages audio input from the microphone, processes it for wake word detection, and launches
    an application and server when the wake word is detected. It also handles cleanup and shutdown.
    """
    def __init__(self):
        """
        Initializes the WakeWordDetector with necessary components: VAD, wake word model,
        audio configurations, and process management attributes.
        """
        self.vad = webrtcvad.Vad(3) # Initialize Voice Activity Detector from webrtcvad, aggressiveness level 3
        self.model = Model(wakeword_models=["Hey_Sen_she_ent.onnx"],
                          inference_framework="onnx") # Load the wake word detection model using OpenWakeWord
        print(f"Loaded models: {self.model.models.keys()}") # Print the loaded wake word models for verification
        self.sample_rate = 16000 # Define the audio sample rate
        self.frame_duration_ms = 80  # Minimum frame duration in milliseconds for phoneme recognition
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)  # Calculate frame size in samples based on sample rate and frame duration
        self.running = True # Flag to control the running state of the detector

        # Audio configuration parameters
        self.pa_format = pyaudio.paInt16 # Define the audio format as 16-bit integers
        self.channels = 1 # Set to mono audio channel
        self.chunk_size = self.sample_rate * 1  # Define chunk size as 1 second of audio samples

        self.buffer_size = int(1 * self.sample_rate) # Buffer size for audio processing, set to 1 second
        self.buffer = np.zeros(self.chunk_size * 2)  # Initialize a buffer with double the chunk size for overlapping audio processing

        self.mic_queue = Queue() # Queue to hold audio frames from the microphone
        self.pa = pyaudio.PyAudio() # Initialize PyAudio for audio input

        # Application configuration
        self.app_path = r"" # Path to the application directory (Electron app and server), needs to be set externally
        self.frontend_process: subprocess.Popen[bytes] | None = None # Process object for the frontend Electron app, initially None
        self.server_process: subprocess.Popen[bytes] | None = None # Process object for the backend server, initially None

    def _audio_input_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> tuple[None, int]:
        """
        PyAudio callback function for non-blocking audio input.
        This function is called by PyAudio whenever there is new audio input data available.
        It puts the audio data into a queue for processing in a separate thread.

        Args:
            in_data (bytes): Audio data as a byte string.
            frame_count (int): Number of frames in the input buffer.
            time_info (dict): Time information related to the audio stream.
            status (int): Status flags.

        Returns:
            tuple[None, int]: Returns (None, pyaudio.paContinue) to continue the audio stream.
        """
        if self.running: # Check if the detector is in running state
            audio_frame = np.frombuffer(in_data, dtype=np.int16) # Convert byte audio data to numpy array of int16
            self.mic_queue.put(audio_frame) # Put the audio frame into the microphone queue
        return (None, pyaudio.paContinue) # Signal PyAudio to continue streaming

    def _mic_thread(self) -> None:
        """
        Thread function dedicated to capturing audio from the microphone using PyAudio in a non-blocking manner.
        It opens an audio stream with a callback function to handle incoming audio data and continuously
        reads from the microphone as long as the detector is running.
        """
        try:
            stream = self.pa.open( # Open a non-blocking audio input stream
                format=self.pa_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback, # Set the callback function to process audio input
                start=True # Start the stream immediately
            )

            while self.running: # Keep the thread alive while the detector is running
                time.sleep(0.01) # Small sleep to reduce CPU usage in the loop

            stream.stop_stream() # Stop the audio stream
            stream.close() # Close the audio stream

        except OSError as e: # Catch potential OSError exceptions, e.g., if microphone is not accessible
            print(f"Microphone error: {e}")

    def _process_audio(self) -> None:
        """
        Main audio processing loop that runs in a separate thread.
        It continuously gets audio frames from the microphone queue, updates the audio buffer,
        performs wake word detection on the buffer, and triggers actions if the wake word is detected.
        """
        while self.running: # Main loop to process audio while detector is running
            try:
                raw_audio = self.mic_queue.get(timeout=0.1) # Get audio frame from queue with a timeout to prevent blocking indefinitely

                # Update buffer with 50% overlap for continuous processing
                self.buffer = np.roll(self.buffer, -len(raw_audio)) # Roll the buffer to make space for new audio data, effectively overlapping previous data
                self.buffer[-len(raw_audio):] = raw_audio # Fill the end of the buffer with the new audio frame

                # Process 1-second windows of audio every 0.5 seconds (due to 50% overlap from chunk_size and buffer update)
                if len(raw_audio) == self.chunk_size: # Process only when a full chunk (1 second) of new audio is received
                    prediction = self.model.predict(self.buffer) # Perform wake word prediction on the current audio buffer
                    print(f"Confidence: {prediction['Hey_Sen_she_ent']:.4f}") # Print the confidence score of the wake word model

                    if prediction["Hey_Sen_she_ent"] > 0.0080: # Check if the confidence score is above a threshold (0.0080)
                        print("\n=== WAKE WORD DETECTED ===")
                        self._trigger_action() # Trigger the action associated with wake word detection
                        time.sleep(2) # Sleep for a short duration after detection to prevent multiple triggers in quick succession

            except Exception as e: # Catch exceptions during audio processing, like queue timeout or model prediction errors
                if str(e) != "": # Avoid printing empty exception strings
                    print(f"Processing error: {e}")

    def _trigger_action(self) -> None:
        """
        Triggers the action to be performed when the wake word is detected.
        In this case, it launches the main application (Electron app and server).
        """
        print("Launching main application...")
        self.launch_app_and_server() # Call the method to launch the application and server

    def launch_app_and_server(self) -> None:
        """
        Launches the Electron frontend application and the Python backend server as subprocesses.
        It first changes the current working directory to the application path, then kills any existing
        Node.js processes to ensure a clean restart, and finally starts the frontend and server using npm and python commands respectively.
        After launching, it waits for the frontend process to complete, and upon completion, it terminates the server process and prepares for restart.
        """
        try:
            os.chdir(self.app_path) # Change the current working directory to the application path

            # Kill any existing Node processes to avoid port conflicts or previous instances
            subprocess.run(['taskkill', '/F', '/IM', 'node.exe'],
                         stderr=subprocess.DEVNULL, # Suppress standard error output from taskkill command
                         check=False) # Do not raise an exception if taskkill fails (e.g., no node processes running)
            sleep(1) # Wait for 1 second to ensure processes are terminated before launching new ones

            # Launch the Electron frontend application using npm start command in a shell
            self.frontend_process = subprocess.Popen('npm run start', shell=True)
            # Launch the Python backend server by running server.py in a shell
            self.server_process = subprocess.Popen('python server.py', shell=True)

            print("App and server launched successfully!")

            # Wait for the frontend process to exit (application closed by user)
            self.frontend_process.wait()

            # When the Electron app closes, stop the backend server process
            if self.server_process: # Check if server process is running before attempting to terminate
                self.server_process.terminate() # Terminate the server process gracefully
            print("App and server closed. Restarting listener...")

            # Reset process variables to None, ready for next wake word detection cycle
            self.frontend_process = None
            self.server_process = None

        except Exception as e: # Catch any exceptions that occur during application launching or process management
            print(f"Error launching app and server: {str(e)}")

    def start(self) -> None:
        """
        Starts the wake word detection process by initiating the microphone and audio processing threads.
        """
        mic_thread = threading.Thread(target=self._mic_thread) # Create a thread for microphone input
        proc_thread = threading.Thread(target=self._process_audio) # Create a thread for audio processing

        mic_thread.start() # Start the microphone thread
        proc_thread.start() # Start the audio processing thread

    def stop(self) -> None:
        """
        Stops the wake word detector gracefully.
        Sets the running flag to False, waits for threads to exit, terminates application processes if running,
        and terminates the PyAudio interface.
        """
        self.running = False # Set the running flag to False to signal threads to stop
        time.sleep(0.5)  # Wait for 0.5 seconds to allow threads to exit gracefully

        # Cleanup application processes if they were launched and are still running
        if self.frontend_process:
            self.frontend_process.terminate() # Terminate the frontend process
        if self.server_process:
            self.server_process.terminate() # Terminate the server process

        self.pa.terminate() # Terminate the PyAudio interface, releasing audio resources

if __name__ == "__main__":
    detector = WakeWordDetector() # Instantiate the WakeWordDetector class

    try:
        print("Listening for 'Hey Sentient'...")
        detector.start() # Start the wake word detection process

        while True: # Keep the main thread alive to listen for KeyboardInterrupt
            time.sleep(0.1)  # Reduced sleep time for more responsive shutdown, checking every 0.1 seconds

    except KeyboardInterrupt: # Handle KeyboardInterrupt (Ctrl+C) to stop the detector gracefully
        print("\nStopping detector...")
        detector.stop() # Call the stop method to perform cleanup and shutdown