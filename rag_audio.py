import speech_recognition as sr
import os
import io
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence

def convert_to_wav(audio_path: str) -> str:
    """
    Convert an audio file to WAV format if it is not already in WAV format.

    This function takes the path to an audio file as input and checks if it is a WAV file.
    If it is not a WAV file, it converts the audio file to WAV format using pydub
    and saves it to a temporary file. If it is already a WAV file, it returns the original path.

    Args:
        audio_path (str): The path to the input audio file.

    Returns:
        str: The path to the WAV file. This will be the original path if the input
             was already a WAV file, or the path to the newly created temporary WAV file.
    """
    # Check if the audio file is already in WAV format
    if audio_path.endswith('.wav'):
        return audio_path

    # If not WAV, convert to WAV using pydub and store in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_path = temp_wav.name # Get the temporary file path
        audio = AudioSegment.from_file(audio_path) # Load the audio file using pydub
        audio.export(temp_path, format="wav") # Export the audio to WAV format to the temporary file

    return temp_path # Return the path to the temporary WAV file

def extract_text_from_audio(
    audio_path: str,
    language: str = "en-US",
    chunk_size: int = 60000  # Default 60 seconds in milliseconds
) -> str:
    """
    Convert speech from an audio file to text using Google Speech Recognition.

    This function takes an audio file, splits it into chunks, and then transcribes each chunk
    using the Google Speech Recognition API. It supports both fixed-size chunking and
    silence-based chunking.

    Args:
        audio_path (str): Path to the audio file that needs to be transcribed.
        language (str, optional): Language code for speech recognition (e.g., 'en-US' for English).
                                   Defaults to "en-US".
        chunk_size (int, optional): Size of audio chunks in milliseconds. If provided, the audio
                                     is split into chunks of this size. If set to 0 or None,
                                     silence-based chunking is used instead. Defaults to 60000 (60 seconds).

    Returns:
        str: Extracted text from the audio. Returns an empty string if no speech is recognized
             or if there are errors during transcription.

    Raises:
        FileNotFoundError: If the audio file specified by `audio_path` does not exist.
        Exception: If any error occurs during the transcription process, such as file reading,
                   audio processing, or speech recognition API errors.
    """
    # Check if the audio file exists at the given path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    try:
        recognizer = sr.Recognizer() # Initialize the speech recognizer
        wav_path = convert_to_wav(audio_path) # Convert the input audio to WAV format if necessary
        audio = AudioSegment.from_wav(wav_path) # Load the WAV audio file using pydub

        chunks = [] # List to store audio chunks
        if chunk_size:
            # Split audio into fixed-size chunks if chunk_size is provided
            for start in range(0, len(audio), chunk_size):
                chunks.append(audio[start:start + chunk_size]) # Append audio chunks of specified size
        else:
            # Split audio based on silence if chunk_size is not provided (or is 0/None)
            chunks = split_on_silence(
                audio,
                min_silence_len=500,      # Minimum length of silence in ms to be considered a silence
                silence_thresh=audio.dBFS - 14, # Consider声音 below -14dBFS relative to the average loudness as silence
                keep_silence=500       # Keep 500ms of silence on both sides of the split
            )

        full_text = [] # List to accumulate text from each chunk

        # Process each audio chunk
        for i, chunk in enumerate(chunks):
            # Export the audio chunk to a temporary in-memory WAV file
            with io.BytesIO() as wav_buffer:
                chunk.export(wav_buffer, format="wav") # Export chunk to WAV format in buffer
                wav_buffer.seek(0) # Rewind the buffer to the beginning

                # Use SpeechRecognition to transcribe the audio chunk from the in-memory WAV file
                with sr.AudioFile(wav_buffer) as source:
                    audio_data = recognizer.record(source) # Record audio data from the in-memory file

                    try:
                        # Recognize speech using Google Speech Recognition
                        text = recognizer.recognize_google(audio_data, language=language)
                        full_text.append(text) # Append recognized text to the list
                    except sr.UnknownValueError:
                        print(f"Could not understand chunk {i}") # Log if speech is not understandable in chunk
                    except sr.RequestError as e:
                        print(f"Speech recognition service error in chunk {i}: {e}") # Log if there's an API request error

        # Clean up: remove the temporary WAV file if it was created during conversion
        if wav_path != audio_path:
            os.remove(wav_path)

        return " ".join(full_text).strip() # Join all transcribed texts and return as a single string

    except Exception as e:
        # Error handling: ensure temporary WAV file is removed even if an error occurs
        if 'wav_path' in locals() and wav_path != audio_path:
            os.remove(wav_path)
        raise Exception(f"Error during transcription: {str(e)}") # Re-raise the exception with a more informative message

# Usage example when the script is run directly
if __name__ == "__main__":
    try:
        # Example usage of the extract_text_from_audio function
        text = extract_text_from_audio(
            "test.mp4", # Path to the audio/video file (replace with your file)
            language="en-US", # Language for speech recognition (English - US)
            chunk_size=60000  # Chunk size of 60 seconds (set to 0 or None to use silence-based splitting)
        )
        print("Extracted Text:", text) # Print the extracted text
    except Exception as e:
        print(f"Error: {str(e)}") # Print any error that occurred during the process