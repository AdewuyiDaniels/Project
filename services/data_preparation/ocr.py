import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
    - image_path: Path to the image file.
    
    Returns:
    - extracted_text: Extracted text from the image.
    """
    # Open the image file
    with Image.open(image_path) as img:
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(img)
    
    return extracted_text
