import speech_recognition as sr
import os
import io
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence

def convert_to_wav(audio_path: str) -> str:

    # Check if the audio file is already in WAV format
    if audio_path.endswith('.wav'):
        return audio_path

    # If not WAV, convert to WAV using pydub and store in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_path = temp_wav.name 
        audio = AudioSegment.from_file(audio_path)
        audio.export(temp_path, format="wav") 

    return temp_path 

def extract_text_from_audio(
    audio_path: str,
    language: str = "en-US",
    chunk_size: int = 60000 
) -> str:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    try:
        recognizer = sr.Recognizer()
        wav_path = convert_to_wav(audio_path)
        audio = AudioSegment.from_wav(wav_path) 

        chunks = []
        if chunk_size:
            # Split audio into fixed-size chunks if chunk_size is provided
            for start in range(0, len(audio), chunk_size):
                chunks.append(audio[start:start + chunk_size])
        else:
            # Split audio based on silence if chunk_size is not provided (or is 0/None)
            chunks = split_on_silence(
                audio,
                min_silence_len=500,     
                silence_thresh=audio.dBFS - 14, 
                keep_silence=500
            )

        full_text = []

        # Process each audio chunk
        for i, chunk in enumerate(chunks):
            # Export the audio chunk to a temporary in-memory WAV file
            with io.BytesIO() as wav_buffer:
                chunk.export(wav_buffer, format="wav")
                wav_buffer.seek(0) # Rewind the buffer to the beginning

                # Use SpeechRecognition to transcribe the audio chunk from the in-memory WAV file
                with sr.AudioFile(wav_buffer) as source:
                    audio_data = recognizer.record(source)

                    try:
                        # Recognize speech using Google Speech Recognition
                        text = recognizer.recognize_google(audio_data, language=language)
                        full_text.append(text)
                    except sr.UnknownValueError:
                        print(f"Could not understand chunk {i}")
                    except sr.RequestError as e:
                        print(f"Speech recognition service error in chunk {i}: {e}") 

        # Clean up: remove the temporary WAV file if it was created during conversion
        if wav_path != audio_path:
            os.remove(wav_path)

        return " ".join(full_text).strip()

    except Exception as e:
        # Error handling: ensure temporary WAV file is removed even if an error occurs
        if 'wav_path' in locals() and wav_path != audio_path:
            os.remove(wav_path)
        raise Exception(f"Error during transcription: {str(e)}") 

if __name__ == "__main__":
    try:
        # Example usage of the extract_text_from_audio function
        text = extract_text_from_audio(
            "test.mp4",
            language="en-US", 
            chunk_size=60000  
        )
        print("Extracted Text:", text) 
    except Exception as e:
        print(f"Error: {str(e)}")