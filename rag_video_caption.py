import yt_dlp
import speech_recognition as sr
import os
import tempfile
import platform
import subprocess
from pathlib import Path
import re
import unicodedata
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Optional

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove or replace characters that are problematic in file systems.

    This function removes or replaces characters that are typically invalid or cause issues
    in filenames across different operating systems. It performs the following operations:
    - Removes characters like <>, :", /, \\, |, ?, *.
    - Replaces spaces with underscores.
    - Normalizes Unicode characters to ASCII, removing accents and other non-ASCII characters.
    - Truncates the filename to a maximum length of 200 characters to avoid path length limitations.

    Args:
        filename (str): The original filename string that needs to be sanitized.

    Returns:
        str: The sanitized filename string, safe for use in file systems.
    """
    # Remove characters that are generally invalid in filenames
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores for better readability and compatibility
    filename = filename.replace(' ', '_')
    # Normalize Unicode characters to ASCII, removing accents and other diacritics
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    # Truncate filename to a maximum of 200 characters to prevent issues with long filenames
    return filename[:200]

def check_ffmpeg() -> str:
    """
    Check if FFmpeg is installed and accessible in the system's PATH, or in common installation directories.

    This function attempts to locate the FFmpeg executable. First, it checks if 'ffmpeg' command is available
    in the system's PATH. If not found, it checks a list of common installation paths for different operating systems
    where FFmpeg might be installed.

    Returns:
        str: The path to the FFmpeg executable if found.

    Raises:
        Exception: If FFmpeg is not found in PATH or common installation directories, an exception is raised
                   informing the user to install FFmpeg and add it to their system's PATH environment variable.
    """
    try:
        # Try to run ffmpeg --version to check if it's in PATH
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'ffmpeg'  # FFmpeg is in PATH, return the command name
    except FileNotFoundError:
        # FFmpeg not in PATH, check common installation directories
        system = platform.system().lower()
        possible_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe', str(Path.home() / 'ffmpeg' / 'bin' / 'ffmpeg.exe'), # User home directory
            '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/opt/homebrew/bin/ffmpeg', # Common Linux/macOS paths
            str(Path.home() / 'ffmpeg' / 'ffmpeg') # User home directory (alternative for macOS/Linux)
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                return path  # Return the full path to ffmpeg executable if found

        # If FFmpeg is not found in PATH or common directories, raise an exception
        raise Exception("FFmpeg not found. Please install it and add it to PATH.")

def download_audio(url: str, output_path: str) -> Optional[str]:
    """
    Download audio from a YouTube video URL using yt-dlp, converting it to WAV format.

    This function uses the yt-dlp library to download the best available audio from a YouTube video
    specified by the URL. It then uses FFmpeg to post-process the downloaded audio, converting it to WAV format.
    The audio file is saved in the specified output path, and the function returns the path to the downloaded WAV file.

    Args:
        url (str): The URL of the YouTube video from which to download audio.
        output_path (str): The directory path where the downloaded audio file should be saved.

    Returns:
        Optional[str]: The full path to the downloaded WAV audio file if successful, otherwise None in case of error.
    """
    try:
        ffmpeg_path = check_ffmpeg() # Ensure FFmpeg is available and get its path
        output_template = os.path.join(output_path, '%(title).100s.%(ext)s') # Output filename template

        ydl_opts = {
            'format': 'bestaudio/best', # Download best available audio format
            'outtmpl': output_template, # Output template for filenames
            'postprocessors': [{ # Post-processing to extract audio and convert to WAV
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav', # Convert to WAV format
                'preferredquality': '192', # Audio quality setting (for codecs that support it)
            }],
            'ffmpeg_location': os.path.dirname(ffmpeg_path) if os.path.dirname(ffmpeg_path) else None, # Specify FFmpeg location for yt-dlp
            'quiet': True, # Suppress yt-dlp console output
            'no_warnings': True, # Suppress yt-dlp warnings
            'restrictfilenames': True # Restrict filenames to only ASCII characters
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting video information...")
            info = ydl.extract_info(url, download=True) # Extract video information and download audio
            safe_title = sanitize_filename(info['title']) # Sanitize title for filename
            audio_file = os.path.join(output_path, f"{safe_title}.wav") # Expected path for the WAV audio file

            # Check if the audio file exists (yt-dlp might sometimes rename or slightly alter filenames)
            if not os.path.exists(audio_file):
                possible_files = [f for f in os.listdir(output_path) if f.endswith('.wav')]
                if possible_files:
                    audio_file = os.path.join(output_path, possible_files[0]) # Use first found WAV file if expected not found

            print(f"Downloaded: {os.path.basename(audio_file)}")
            return audio_file # Return path to the downloaded audio file
    except Exception as e:
        print(f"\nError downloading audio: {str(e)}")
        return None # Return None if download fails

def transcribe_audio(audio_path: str, language: str = "en-US", chunk_size: int = 60000) -> Optional[str]:
    """
    Transcribe audio from a WAV file using Google Speech Recognition API, processing audio in chunks.

    This function takes a path to a WAV audio file, splits it into chunks of specified size, and then
    uses the SpeechRecognition library to transcribe each chunk using Google's speech recognition service.
    It concatenates the transcribed text from all chunks and returns the full transcript.

    Args:
        audio_path (str): The path to the WAV audio file to transcribe.
        language (str, optional): The language code for speech recognition (e.g., 'en-US' for English).
                                  Defaults to "en-US".
        chunk_size (int, optional): The size of audio chunks in milliseconds to process. Defaults to 60000 (60 seconds).

    Returns:
        Optional[str]: The transcribed text if successful, otherwise None if transcription fails or audio file is not found.
    """
    if not os.path.exists(audio_path):
        print("Audio file not found")
        return None # Return None if audio file does not exist

    recognizer = sr.Recognizer() # Initialize speech recognizer
    audio = AudioSegment.from_wav(audio_path) # Load audio file using pydub

    # Split audio into chunks of specified size
    chunks = [audio[start:start + chunk_size] for start in range(0, len(audio), chunk_size)]
    full_text = [] # Initialize list to store transcribed text chunks

    for i, chunk in enumerate(chunks):
        # Create a temporary WAV file for each chunk to avoid issues with file access/permissions
        temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav") # Create a temporary file
        os.close(temp_fd)  # Immediately close the file descriptor, only need the path

        try:
            chunk.export(temp_wav_path, format="wav") # Export audio chunk to temporary WAV file

            with sr.AudioFile(temp_wav_path) as source: # Open temporary WAV file as audio source
                audio_data = recognizer.listen(source) # Listen to audio data from source
                try:
                    text = recognizer.recognize_google(audio_data, language=language) # Recognize speech using Google API
                    full_text.append(text) # Append transcribed text to the list
                except sr.UnknownValueError:
                    print(f"Could not understand chunk {i}") # Log if speech is unintelligible
                except sr.RequestError as e:
                    print(f"Speech recognition service error in chunk {i}: {e}") # Log API request errors

        finally:
            os.remove(temp_wav_path)  # Ensure temporary file is always deleted after processing

    return " ".join(full_text).strip() # Join all transcribed chunks, remove leading/trailing spaces, and return

def generate_transcript(url: str) -> Optional[str]:
    """
    Generate a full transcript from a YouTube video URL.

    This function orchestrates the process of generating a transcript:
    1. Creates a temporary directory to store downloaded audio.
    2. Downloads audio from the YouTube video URL using `download_audio`.
    3. Transcribes the downloaded audio file using `transcribe_audio`.
    4. Cleans up the temporary directory after processing.

    Args:
        url (str): The URL of the YouTube video to transcribe.

    Returns:
        Optional[str]: The generated transcript text if successful, otherwise None in case of errors during download or transcription.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir: # Create a temporary directory that will be automatically cleaned up
            print("Downloading audio...")
            audio_file = download_audio(url, temp_dir) # Download audio to the temporary directory

            if not audio_file or not os.path.exists(audio_file):
                print("Failed to download audio file")
                return None # Return None if audio download failed

            print("Generating transcript...")
            transcript = transcribe_audio(audio_file) # Transcribe the downloaded audio file

            if transcript:
                print("\nTranscript Generated Successfully!")
                return transcript # Return the generated transcript
            else:
                print("\nNo transcript generated.")
                return None # Return None if transcription failed
    except Exception as e:
        print(f"Error generating transcript: {str(e)}")
        return None # Return None if any error occurred

def save_transcript(transcript: str, output_file: str) -> None:
    """
    Save the given transcript text to a specified output file.

    This function writes the transcript text to a file, using UTF-8 encoding to support a wide range of characters.

    Args:
        transcript (str): The transcript text to be saved.
        output_file (str): The path to the file where the transcript should be saved.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f: # Open the output file in write mode with UTF-8 encoding
            f.write(transcript) # Write the transcript to the file
        print(f"Transcript saved to {output_file}") # Inform user where the transcript is saved
    except Exception as e:
        print(f"Error saving transcript: {str(e)}") # Print error message if saving fails

def verify_setup() -> bool:
    """
    Verify that the necessary setup components (like FFmpeg) are working correctly.

    This function checks if FFmpeg is installed and accessible, and attempts to run a version check command
    to ensure it is functioning as expected.

    Returns:
        bool: True if the setup verification is successful, False otherwise.
    """
    try:
        ffmpeg_path = check_ffmpeg() # Check if FFmpeg is found and get its path
        print(f"✓ FFmpeg found at: {ffmpeg_path}")
        subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True) # Run FFmpeg version command to verify
        print("✓ FFmpeg is working properly")
        return True # Return True if FFmpeg check is successful
    except Exception as e:
        print(f"Setup verification failed: {str(e)}")
        return False # Return False if setup verification failed

def main() -> None:
    """
    Main function to run the YouTube to transcript generation process.

    This function guides the user through the process of:
    1. Verifying the setup (FFmpeg).
    2. Prompting for a YouTube video URL.
    3. Generating a transcript from the video.
    4. Prompting for an output filename and saving the transcript to a file.
    """
    print("Verifying setup...")
    if not verify_setup(): # Verify setup before proceeding
        print("\nPlease fix the setup issues and try again.")
        return # Exit if setup verification fails

    print("\nSetup verified successfully!")

    url = input("\nEnter YouTube video URL: ") # Prompt user to enter YouTube URL
    transcript = generate_transcript(url) # Generate transcript from the given URL

    if transcript: # If transcript was successfully generated
        output_file = input("Enter output filename (e.g. transcript.txt): ") # Prompt for output filename
        save_transcript(transcript, output_file) # Save the transcript to the specified file
    else:
        print("Failed to generate transcript") # Inform user if transcript generation failed

if __name__ == "__main__":
    main() # Execute main function when script is run directly