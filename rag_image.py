import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import os
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans

@dataclass
class TextDetectionResult:
    """
    Dataclass to store the result of text detection.

    Attributes:
        text (str): The extracted text.
        confidence (float): Confidence score of the extracted text (0 to 1).
        method (str): The OCR method used to extract the text (e.g., 'tesseract', 'easyocr').
    """
    text: str
    confidence: float
    method: str

class AdvancedSceneTextReader:
    """
    A class for advanced scene text recognition using multiple OCR engines and image processing techniques.

    This class integrates Tesseract OCR and EasyOCR to extract text from images.
    It employs various image preprocessing steps like noise removal, contrast enhancement,
    and binary masking to improve OCR accuracy. It also analyzes image characteristics
    to adaptively apply different text extraction strategies.
    """
    def __init__(self, tesseract_path: str = r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        """
        Initialize the AdvancedSceneTextReader with OCR engines and configurations.

        Args:
            tesseract_path (str, optional): Path to the Tesseract OCR executable.
                Defaults to r'C:\Program Files\Tesseract-OCR\tesseract.exe'.
        """
        self.tesseract_path = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path # Set Tesseract command path
        self.reader = easyocr.Reader(['en']) # Initialize EasyOCR reader for English language
        self.logger = logging.getLogger(__name__) # Initialize logger for this class

    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze the characteristics of an image to determine properties like brightness, contrast, and texture.

        These characteristics can be used to adapt image processing and OCR strategies.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            Dict[str, float]: A dictionary containing analysis results, including:
                'brightness' (float): Average brightness of the image.
                'contrast' (float): Contrast of the image.
                'texture' (float): Measure of texture in the image.
        """
        # Convert the image to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the average brightness of the grayscale image
        avg_brightness = np.mean(gray)

        # Calculate the standard deviation of pixel intensities to estimate contrast
        contrast = np.std(gray)

        # Detect texture by comparing the grayscale image with its blurred version
        texture_kernel = np.ones((5,5),np.float32)/25 # Define a kernel for blurring
        blurred = cv2.filter2D(gray, -1, texture_kernel) # Apply blurring
        texture_measure = np.mean(np.absolute(gray - blurred)) # Measure texture as mean absolute difference

        return {
            'brightness': float(avg_brightness), # Return brightness as float
            'contrast': float(contrast),       # Return contrast as float
            'texture': float(texture_measure)    # Return texture measure as float
        }

    def _create_binary_masks(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple binary masks from the input image using different thresholding and segmentation techniques.

        These masks are used to enhance text visibility for OCR engines by isolating text regions.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            List[np.ndarray]: A list of binary masks, each as a NumPy array (0 and 255 values).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
        masks: List[np.ndarray] = [] # Initialize list to store binary masks

        # 1. Basic thresholding using Otsu's method to automatically determine threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(binary) # Add basic binary mask

        # 2. Adaptive thresholding: thresholding is applied to smaller regions of the image, adapting to varying lighting conditions
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        masks.append(adaptive) # Add adaptive binary mask

        # 3. Color-based segmentation using K-Means clustering in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convert image to LAB color space
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # Initialize KMeans clustering with 2 clusters
        h, w = gray.shape # Get height and width of grayscale image
        reshaped_lab = lab.reshape((h * w, 3)) # Reshape LAB image to be a list of pixels
        kmeans.fit(reshaped_lab) # Fit KMeans on reshaped LAB image
        segmented = kmeans.labels_.reshape((h, w)) * 255 # Create segmentation mask from cluster labels
        masks.append(segmented.astype(np.uint8)) # Add color-based segmentation mask

        return masks # Return the list of created binary masks

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the contrast of the input image using Contrast Limited Adaptive Histogram Equalization (CLAHE).

        CLAHE improves local contrast and reveals more details in both dark and bright regions of the image.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Contrast-enhanced image as a NumPy array.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convert image to LAB color space
        l, a, b = cv2.split(lab) # Split LAB image into L, A, B channels
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Initialize CLAHE with clip limit and tile grid size
        cl = clahe.apply(l) # Apply CLAHE to the L channel (lightness)
        enhanced = cv2.merge((cl,a,b)) # Merge the CLAHE-processed L channel with original A and B channels
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR) # Convert back to BGR color space and return

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the input image using a bilateral filter while preserving edges.

        Bilateral filtering is effective in noise reduction while keeping edges sharp, which is important for text recognition.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Denoised image as a NumPy array.
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75) # Parameters: image, kernel size, color sigma, space sigma
        return denoised # Return the denoised image

    def _extract_text_tesseract(self, image: np.ndarray, config: str = '') -> TextDetectionResult:
        """
        Extract text from an image using Tesseract OCR engine.

        It also calculates a confidence score for the extracted text based on Tesseract's output.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            config (str, optional): Additional configuration string for Tesseract. Defaults to ''.

        Returns:
            TextDetectionResult: A TextDetectionResult object containing the extracted text, confidence score, and method name ('tesseract').
                                Returns an empty TextDetectionResult if text extraction fails or no text is found.
        """
        try:
            # Use pytesseract to perform OCR and get detailed data output
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            text_parts: List[str] = [] # List to store individual text components
            confidence_scores: List[int] = [] # List to store confidence scores for each text component

            # Iterate through the OCR data to filter out non-text and collect text parts and confidence scores
            for i, conf in enumerate(data['conf']):
                if conf > -1:  # Confidence value of -1 indicates non-text or noise
                    text_parts.append(data['text'][i]) # Append extracted text part
                    confidence_scores.append(int(conf)) # Append confidence score, converting to integer

            # If text parts are found, calculate average confidence and return result
            if text_parts:
                avg_confidence = sum(confidence_scores) / len(confidence_scores) # Calculate average confidence
                return TextDetectionResult(
                    text=' '.join(text_parts).strip(), # Join text parts into a single string and strip whitespace
                    confidence=avg_confidence / 100, # Normalize confidence to a 0-1 scale
                    method='tesseract' # Indicate that Tesseract was used
                )
        except Exception as e:
            self.logger.warning(f"Tesseract error: {str(e)}") # Log any exceptions during Tesseract OCR

        return TextDetectionResult('', 0.0, 'tesseract') # Return empty result if text extraction failed

    def _extract_text_easyocr(self, image: np.ndarray) -> TextDetectionResult:
        """
        Extract text from an image using EasyOCR engine.

        It also calculates an average confidence score for the extracted text from EasyOCR's detections.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            TextDetectionResult: A TextDetectionResult object containing the extracted text, confidence score, and method name ('easyocr').
                                Returns an empty TextDetectionResult if text extraction fails or no text is found.
        """
        try:
            # Use EasyOCR reader to detect and recognize text in the image
            results: List[Tuple[list, str, float]] = self.reader.readtext(image) # results is a list of (bounding_box, text, confidence)
            if results:
                text_parts: List[str] = [] # List to store extracted text parts
                confidence_scores: List[float] = [] # List to store confidence scores

                # Aggregate text parts and confidence scores from EasyOCR results
                for detection in results:
                    text_parts.append(detection[1]) # Append the detected text
                    confidence_scores.append(detection[2]) # Append the confidence score

                # Return TextDetectionResult with joined text and average confidence
                return TextDetectionResult(
                    text=' '.join(text_parts).strip(), # Join text parts and strip whitespace
                    confidence=sum(confidence_scores) / len(confidence_scores), # Calculate average confidence
                    method='easyocr' # Method used is EasyOCR
                )
        except Exception as e:
            self.logger.warning(f"EasyOCR error: {str(e)}") # Log any exceptions during EasyOCR processing

        return TextDetectionResult('', 0.0, 'easyocr') # Return empty result on failure

    def extract_text(self, image_path: str) -> str:
        """
        Main method to extract text from an image using a combination of OCR techniques and image processing.

        It applies preprocessing steps, uses both Tesseract and EasyOCR engines on different image versions,
        and selects the best result based on confidence and text length.

        Args:
            image_path (str): Path to the image file from which text needs to be extracted.

        Returns:
            str: Extracted text from the image. Returns an empty string if no text could be reliably extracted.

        Raises:
            FileNotFoundError: If the image file at `image_path` does not exist.
            ValueError: If the image file cannot be read by OpenCV.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}") # Raise error if image file does not exist

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read image file") # Raise error if image reading fails

        # Analyze image characteristics to potentially adapt processing steps (currently not used but can be extended)
        characteristics = self._analyze_image_characteristics(image)

        results: List[TextDetectionResult] = [] # Initialize list to store text extraction results

        # 1. Process the original image with contrast enhancement and noise removal
        enhanced = self._enhance_contrast(image) # Enhance contrast of the image
        denoised = self._remove_noise(enhanced) # Remove noise from the enhanced image

        # 2. Create binary masks to potentially isolate text regions
        masks = self._create_binary_masks(denoised) # Generate binary masks

        # 3. Apply OCR on each binary mask
        for mask in masks:
            # Try both Tesseract and EasyOCR on each mask
            results.append(self._extract_text_tesseract(mask)) # Extract text using Tesseract on the mask
            results.append(self._extract_text_easyocr(mask)) # Extract text using EasyOCR on the mask

        # 4. Also apply OCR on the denoised and enhanced image itself
        results.append(self._extract_text_tesseract(denoised)) # Tesseract on denoised image
        results.append(self._extract_text_easyocr(denoised)) # EasyOCR on denoised image

        # 5. Filter out results with empty text and select the best result
        valid_results = [r for r in results if r.text.strip()] # Filter out results with empty text
        if not valid_results:
            return "" # Return empty string if no valid text results

        # Select the best result based on confidence score and length of extracted text
        # Prioritizes higher confidence and then longer text as better result
        best_result = max(valid_results,
                         key=lambda x: (x.confidence, len(x.text.strip()))) # Find result with max confidence and text length

        return best_result.text.strip() # Return the text from the best result, stripping any leading/trailing whitespace

def main() -> None:
    """
    Main function to demonstrate the AdvancedSceneTextReader functionality.

    It prompts the user for an image path, extracts text from the image, and prints the extracted text to the console.
    Handles potential exceptions during the process and prints error messages if necessary.
    """
    try:
        reader = AdvancedSceneTextReader() # Initialize the AdvancedSceneTextReader

        # Prompt user to enter the path to the image file
        image_path = input("Enter the path to your image file: ")

        # Extract text from the image using the reader
        extracted_text = reader.extract_text(image_path)

        # Check if any text was extracted and print the result
        if not extracted_text:
            print("No text was extracted from the image.") # Inform user if no text was extracted
        else:
            print("\nExtracted Text:")
            print("--------------")
            print(extracted_text) # Print the extracted text

    except Exception as e:
        print(f"Error: {str(e)}") # Print any error that occurred during the process

if __name__ == "__main__":
    main() # Execute main function when script is run directly