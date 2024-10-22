import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
import re

def preprocess_image(image):
    preprocessed_images = []
    
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply various thresholding techniques
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    morph = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)
    
    # Denoise
    denoised = cv.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Add all processed images to the list
    preprocessed_images.extend([gray, binary, adaptive, morph, denoised])
    
    return preprocessed_images

def extract_text_tesseract(image):
    pil_image = Image.fromarray(image)
    
    # Try different OCR configurations
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 4',
        '--oem 3 --psm 11'
    ]
    
    results = []
    for config in configs:
        text = pytesseract.image_to_string(pil_image, lang='eng', config=config)
        results.append(text)
    
    return results

def clean_text(text):
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned

def filter_relevant_text(texts):
    relevant_words = ['wheat', 'pasta', 'spaghetti', 'noodle', 'macaroni','turmeric','powder','dark','fantasy', 'penne','saffola','oats','kwality', 'choco', 'flakes','kellogs','chocos','Tedhe','medhe','good','day','fortune','suji','kohinoor','matic','Diaper', 'pants','baby', 'pants','Aloo',' Bhujia','TEDHE',' MEDHE','Bourn',' Vita','Cadbury ','Choco ','Chips','Raw','Peanut','comfort','Cuddles','super','pants','Dettol','Dove','Ezee','fab','Godrej','Kachi ghani','mustard oil','sugar','suji','Mix',' fruit','ghadi','Detergent','happy','happy','Creame','Sandwiches','huggies','ice','popz', 'Basmati ','Rice','Lays','Levista','Coffee','LUX','Margo','Surf','excel','Stains','Real','Fruit','juice',"NACHO",'POTATO','CHIPS','PEAS']
    filtered_texts = []
    
    for text in texts:
        words = text.lower().split()
        if any(word in relevant_words for word in words):
            filtered_texts.append(text)
    
    return filtered_texts

def process_image(image_path):
    # Load the image
    img = cv.imread(image_path)
    if img is None:
        return "Error: Unable to read the image file."

    # Preprocess the image
    preprocessed_images = preprocess_image(img)

    all_results = []
    for processed_img in preprocessed_images:
        texts = extract_text_tesseract(processed_img)
        all_results.extend(texts)

    # Clean and filter the results
    cleaned_results = [clean_text(text) for text in all_results]
    filtered_results = filter_relevant_text(cleaned_results)

    # Remove duplicates and join result
    unique_results = list(set(filtered_results))
    return ' | '.join(unique_results) if unique_results else "No relevant text extracted"

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\lenovo\OneDrive\Pictures\Screenshots\goodday.jpg"# Replace with your image path
    result = process_image(image_path)
    print("Extracted Text:")
    print(result)
