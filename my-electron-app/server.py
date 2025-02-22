import asyncio
import websockets
import speech_recognition as sr
import io
import base64

async def process_audio(websocket: websockets.WebSocketServerProtocol):
    """
    Processes audio data received through a WebSocket connection, performs speech recognition,
    and sends the transcribed text back to the client.

    Args:
        websocket: A WebSocketServerProtocol instance representing the connection to the client.

    Returns:
        None. Sends the transcription or an error message back to the client via the websocket.
    """
    recognizer = sr.Recognizer() # Initialize speech recognizer from speech_recognition library

    async for message in websocket: # Asynchronously iterate over messages received from the websocket
        try:
            # Decode Base64 audio data received from the client
            audio_data: bytes = base64.b64decode(message)
            # Wrap the decoded audio bytes in an io.BytesIO object to simulate a file
            audio_file = io.BytesIO(audio_data)

            # Use SpeechRecognition library to read the audio file
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source) # Record audio data from the source

            # Perform speech recognition using Google Web Speech API
            text: str = recognizer.recognize_google(audio)
            await websocket.send(text)  # Send the transcribed text back to the client via the websocket
        except Exception as e:
            # If any error occurs during processing, send an error message back to the client
            await websocket.send(f"Error: {str(e)}")

async def main():
    """
    Sets up and starts a WebSocket server that listens for audio data for speech recognition.

    The server runs on 'localhost' and port 8765. It uses the 'process_audio' function
    to handle incoming WebSocket connections and process audio messages.

    Returns:
        None. This function runs indefinitely, keeping the WebSocket server alive.
    """
    # Start a WebSocket server, binding the 'process_audio' function to handle connections
    server = await websockets.serve(process_audio, "localhost", 8765)  # Correct handler, process_audio will handle each websocket connection
    print("WebSocket server running on ws://localhost:8765")
    await server.wait_closed()  # Keep the server running until it is manually closed

if __name__ == "__main__":
    # Entry point of the script: runs the main function to start the WebSocket server
    asyncio.run(main())  # Initializes the asyncio event loop and runs the main coroutine