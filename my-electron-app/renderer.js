/**
 * @file script.js
 * @description This script handles audio recording from the microphone, sends it to a WebSocket server for transcription,
 * and implements text-to-speech functionality using the browser's SpeechSynthesis API.
 */

/**
 * @type {WebSocket}
 * @description WebSocket connection to the transcription server.
 */
const socket = new WebSocket("ws://localhost:8765");

/**
 * @event WebSocket#onmessage
 * @description Event listener for incoming messages from the WebSocket server.
 * Updates the transcription text on the webpage with the received data.
 * @param {MessageEvent} event - The message event containing data from the server.
 */
socket.onmessage = (event) => {
	console.log("Transcribed text:", event.data);
	/**
	 * @type {HTMLElement}
	 * @description DOM element to display the transcribed text.
	 */
	const transcriptionElement = document.getElementById("transcription");
	if (transcriptionElement) {
		transcriptionElement.innerText = event.data; // Update the text content of the transcription element with received data.
	}
};

/**
 * @async
 * @function startRecording
 * @description Initiates audio recording from the user's microphone, processes the audio,
 * and sends it to the WebSocket server for speech-to-text transcription.
 * The recording lasts for 5 seconds.
 * @returns {Promise<void>}
 */
async function startRecording() {
	/**
	 * @type {MediaStream}
	 * @description Stream containing audio data from the microphone.
	 */
	const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
	/**
	 * @type {MediaRecorder}
	 * @description MediaRecorder instance to record audio from the stream.
	 */
	const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" }); // Initialize MediaRecorder with audio stream and MIME type.

	/**
	 * @type {BlobPart[]}
	 * @description Array to store audio chunks recorded by MediaRecorder.
	 */
	let audioChunks = [];
	/**
	 * @event MediaRecorder#ondataavailable
	 * @description Event listener for when audio data is available from MediaRecorder.
	 * Pushes the received audio data (BlobEvent) into the audioChunks array.
	 * @param {BlobEvent} event - The BlobEvent containing audio data.
	 */
	mediaRecorder.ondataavailable = (event) => {
		audioChunks.push(event.data); // Add received audio data to the audioChunks array.
	};

	/**
	 * @event MediaRecorder#onstop
	 * @async
	 * @description Event listener for when MediaRecorder stops recording.
	 * Processes the recorded audio chunks, converts them to WAV format,
	 * and sends the WAV data as Base64 string to the WebSocket server.
	 */
	mediaRecorder.onstop = async () => {
		/**
		 * @type {Blob}
		 * @description Blob containing all recorded audio chunks as WebM format.
		 */
		const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
		/**
		 * @type {ArrayBuffer}
		 * @description ArrayBuffer representation of the audio Blob.
		 */
		const audioBuffer = await audioBlob.arrayBuffer();

		// Convert WebM to WAV using Web Audio API
		/**
		 * @type {AudioContext}
		 * @description AudioContext instance for Web Audio API operations.
		 */
		const audioContext = new AudioContext();
		/**
		 * @type {AudioBuffer}
		 * @description AudioBuffer decoded from the audio ArrayBuffer (WebM).
		 */
		const audioData = await audioContext.decodeAudioData(audioBuffer);
		/**
		 * @type {Blob}
		 * @description WAV Blob encoded from the AudioBuffer.
		 */
		const wavBlob = encodeWAV(audioData); // Convert AudioBuffer to WAV Blob using encodeWAV function.

		// Convert WAV to Base64 and send to WebSocket
		/**
		 * @type {FileReader}
		 * @description FileReader to read the WAV Blob as Data URL.
		 */
		const reader = new FileReader();
		/**
		 * @event FileReader#onloadend
		 * @description Event listener for when FileReader finishes reading the WAV Blob.
		 * Extracts Base64 data from the Data URL and sends it to the WebSocket server.
		 */
		reader.onloadend = () => {
			/**
			 * @type {string}
			 * @description Base64 encoded audio data extracted from the Data URL.
			 */
			const base64Audio = reader.result.split(",")[1]; // Extract Base64 data part from Data URL.
			socket.send(base64Audio); // Send Base64 audio data to WebSocket server.
		};
		reader.readAsDataURL(wavBlob); // Read WAV Blob as Data URL to trigger onloadend event.
	};

	mediaRecorder.start(); // Start recording audio.

	setTimeout(() => {
		mediaRecorder.stop(); // Stop recording after 5 seconds.
	}, 5000); // Set timeout to stop recording after 5 seconds.
}

/**
 * @function encodeWAV
 * @description Encodes an AudioBuffer into a WAV Blob.
 * @param {AudioBuffer} audioBuffer - The AudioBuffer to encode.
 * @returns {Blob} - WAV Blob containing the encoded audio data.
 */
function encodeWAV(audioBuffer) {
	/**
	 * @type {number}
	 * @description Number of channels in the audio buffer.
	 */
	const numOfChan = audioBuffer.numberOfChannels;
	/**
	 * @type {number}
	 * @description Sample rate of the audio buffer.
	 */
	const sampleRate = audioBuffer.sampleRate;
	/**
	 * @type {number}
	 * @description Number of frames (samples per channel) in the audio buffer.
	 */
	const numFrames = audioBuffer.length;
	/**
	 * @type {ArrayBuffer}
	 * @description ArrayBuffer to hold WAV file data. Size calculated based on header and audio data.
	 */
	let wavBuffer = new ArrayBuffer(44 + numFrames * numOfChan * 2); // WAV header (44 bytes) + audio data.
	/**
	 * @type {DataView}
	 * @description DataView to write binary data to the ArrayBuffer.
	 */
	let view = new DataView(wavBuffer);

	// RIFF Header
	writeUTFBytes(view, 0, "RIFF"); // Chunk ID
	view.setUint32(4, 36 + numFrames * numOfChan * 2, true); // Chunk Size (remaining file size)
	writeUTFBytes(view, 8, "WAVE"); // Format

	// fmt subchunk
	writeUTFBytes(view, 12, "fmt "); // Subchunk1 ID
	view.setUint32(16, 16, true); // Subchunk1 Size (16 for PCM)
	view.setUint16(20, 1, true); // Audio Format (1 for PCM)
	view.setUint16(22, numOfChan, true); // Number of Channels
	view.setUint32(24, sampleRate, true); // Sample Rate
	view.setUint32(28, sampleRate * numOfChan * 2, true); // Byte Rate (Sample Rate * Num Channels * Bits per Sample/8)
	view.setUint16(32, numOfChan * 2, true); // Block Align (Num Channels * Bits per Sample/8)
	view.setUint16(34, 16, true); // Bits per Sample (16 bits)

	// data subchunk
	writeUTFBytes(view, 36, "data"); // Subchunk2 ID
	view.setUint32(40, numFrames * numOfChan * 2, true); // Subchunk2 Size (data size)

	// Write audio samples
	let offset = 44; // Offset after header
	for (let i = 0; i < numFrames; i++) {
		for (let channel = 0; channel < numOfChan; channel++) {
			/**
			 * @type {number}
			 * @description Audio sample value, scaled and clipped to 16-bit range.
			 */
			let sample = audioBuffer.getChannelData(channel)[i] * 32767; // Scale to 16-bit signed integer range.
			view.setInt16(offset, sample, true); // Write 16-bit sample to DataView (little-endian).
			offset += 2; // Increment offset by 2 bytes (size of int16).
		}
	}

	return new Blob([view], { type: "audio/wav" }); // Create WAV Blob from DataView and return.
}

/**
 * @function writeUTFBytes
 * @description Helper function to write a UTF-8 string into a DataView at a specified offset.
 * @param {DataView} view - The DataView to write to.
 * @param {number} offset - The starting offset in bytes.
 * @param {string} string - The string to write.
 * @returns {void}
 */
function writeUTFBytes(view, offset, string) {
	for (let i = 0; i < string.length; i++) {
		view.setUint8(offset + i, string.charCodeAt(i)); // Write each character's UTF-8 code unit as a byte.
	}
}

/**
 * @type {HTMLElement | null}
 * @description Button element to start audio recording.
 */
const startButton = document.getElementById("startButton");
if (startButton) {
	startButton.addEventListener("click", startRecording); // Attach event listener to startButton to trigger startRecording function.
}

/**
 * @type {SpeechSynthesisUtterance | null}
 * @description Instance of SpeechSynthesisUtterance to control text-to-speech. Initialized to null.
 */
let utterance = null; // Store speech instance
/**
 * @type {SpeechSynthesisVoice[]}
 * @description Array to store available speech synthesis voices. Initialized as empty array.
 */
let voices = []; // Store available voices

/**
 * @function loadVoices
 * @description Loads available speech synthesis voices from the browser and populates the voice selection dropdown.
 * @returns {void}
 */
function loadVoices() {
	voices = speechSynthesis.getVoices(); // Get all available voices from speechSynthesis API.
	/**
	 * @type {HTMLElement | null}
	 * @description Select dropdown element for voice selection.
	 */
	const voiceSelect = document.getElementById("voiceSelect");
	if (voiceSelect) {
		// Clear existing options
		voiceSelect.innerHTML = ""; // Clear any existing options in the voiceSelect dropdown.

		// Populate dropdown with voices
		voices.forEach((voice, index) => {
			/**
			 * @type {HTMLOptionElement}
			 * @description Option element created for each voice in the dropdown.
			 */
			const option = document.createElement("option");
			option.value = index; // Set option value to the index of the voice in the voices array.
			option.textContent = `${voice.name} (${voice.lang})`; // Set option text to voice name and language.
			voiceSelect.appendChild(option); // Add the option to the voiceSelect dropdown.
		});
	}
}

/**
 * @function speakText
 * @description Converts given text to speech using SpeechSynthesis API with selected voice and settings.
 * @param {string} text - The text to be spoken.
 * @returns {void}
 */
function speakText(text) {
	if (!speechSynthesis.speaking) {
		// Ensure no overlapping speech by checking if speechSynthesis is currently speaking.
		utterance = new SpeechSynthesisUtterance(text); // Create a new SpeechSynthesisUtterance instance with the given text.
		/**
		 * @type {HTMLSelectElement | null}
		 * @description Select dropdown element for voice selection.
		 */
		const voiceSelectElement = document.getElementById("voiceSelect");
		if (voiceSelectElement) {
			const selectedVoiceIndex = voiceSelectElement.value; // Get the selected voice index from the dropdown.
			utterance.voice = voices[selectedVoiceIndex]; // Set the voice for the utterance based on selected index.
			utterance.lang = voices[selectedVoiceIndex].lang; // Set the language for the utterance based on selected voice.
		}
		utterance.rate = 1; // Speed (1 = normal)
		utterance.pitch = 1; // Pitch (1 = normal)

		utterance.onend = () => console.log("Speech complete"); // Log a message when speech finishes.
		speechSynthesis.speak(utterance); // Start speaking the utterance.
	}
}

/**
 * @type {HTMLElement | null}
 * @description Button element to trigger text-to-speech.
 */
const speakButton = document.getElementById("speakButton");
if (speakButton) {
	speakButton.addEventListener("click", () => {
		// Add event listener to speakButton.
		/**
		 * @type {HTMLInputElement | null}
		 * @description Input element for text-to-speech input.
		 */
		const textInput = document.getElementById("ttsInput");
		if (textInput) {
			const text = textInput.value; // Get the text from the ttsInput element.
			if (text.trim()) {
				speakText(text); // Call speakText function to convert text to speech if text is not empty.
			} else {
				alert("Please enter some text!"); // Alert user if no text is entered.
			}
		}
	});
}

/**
 * @type {HTMLElement | null}
 * @description Button element to pause text-to-speech.
 */
const pauseButton = document.getElementById("pauseButton");
if (pauseButton) {
	pauseButton.addEventListener("click", () => {
		// Add event listener to pauseButton.
		if (speechSynthesis.speaking && !speechSynthesis.paused) {
			// Check if speech is active and not already paused.
			speechSynthesis.pause(); // Pause the speech synthesis.
			console.log("Speech paused"); // Log message to console.
		}
	});
}

/**
 * @type {HTMLElement | null}
 * @description Button element to resume text-to-speech.
 */
const resumeButton = document.getElementById("resumeButton");
if (resumeButton) {
	resumeButton.addEventListener("click", () => {
		// Add event listener to resumeButton.
		if (speechSynthesis.paused) {
			// Check if speech synthesis is paused.
			speechSynthesis.resume(); // Resume the speech synthesis.
			console.log("Speech resumed"); // Log message to console.
		}
	});
}

// Ensure voices are loaded properly
speechSynthesis.onvoiceschanged = loadVoices; // Attach event listener to speechSynthesis.onvoiceschanged to load voices when voices are changed (e.g., voice installation).
loadVoices(); // Initial call to load voices when the script is loaded.