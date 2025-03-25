import pytesseract
from PIL import Image
import easyocr

# Try using Tesseract, fallback to EasyOCR if unavailable
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
reader = easyocr.Reader(['en'])

def extract_text(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            raise ValueError("No readable text found with Tesseract.")
        return text.strip()
    except Exception:
        try:
            results = reader.readtext(image_path)
            return " ".join([res[1] for res in results]) or "No readable text found."
        except Exception as e:
            return f"Error extracting text: {str(e)}"

