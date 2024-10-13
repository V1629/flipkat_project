import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
import re

def preprocess_image(image):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Denoise
    denoised = cv.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    return [gray, binary, denoised]

def extract_text_with_positions(image):
    pil_image = Image.fromarray(image)
    ocr_data = pytesseract.image_to_data(pil_image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    
    text_with_positions = []
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i]
        if int(ocr_data['conf'][i]) > 60 and len(word) > 1:
            x, y = ocr_data['left'][i], ocr_data['top'][i]
            text_with_positions.append((word, x, y))
    
    return text_with_positions

def order_words_by_position(words_with_positions):
    sorted_words = sorted(words_with_positions, key=lambda w: (w[2], w[1]))
    ordered_text = [word[0] for word in sorted_words]
    return ordered_text

def clean_text(text):
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned = ' '.join(cleaned.split())
    return cleaned

def remove_duplicates(ordered_text):
    unique_words = []
    seen = set()
    for word in ordered_text:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    return unique_words

def process_image(image):
    preprocessed_images = preprocess_image(image)
    
    all_results = []
    for processed_img in preprocessed_images:
        text_with_positions = extract_text_with_positions(processed_img)
        all_results.extend(text_with_positions)
    
    ordered_words = order_words_by_position(all_results)
    unique_ordered_words = remove_duplicates(ordered_words)
    final_result = ' '.join([clean_text(word) for word in unique_ordered_words])
    
    return final_result if final_result else "No relevant words found"

def extract_product_info(text):
    brand = re.search(r'(Miss & Chief|Himalaya|Bikano|Bingo)', text, re.IGNORECASE)
    brand = brand.group(1) if brand else "Unknown"
    
    product_type = re.search(r'(Diaper Pants|baby pants|Aloo Bhujia|Tedhe Medhe)', text, re.IGNORECASE)
    product_type = product_type.group(1) if product_type else "Unknown"
    
    size = re.search(r'XL', text)
    size = size.group() if size else "Unknown"
    
    count = re.search(r'\d+\s*(pants|pcs)', text, re.IGNORECASE)
    count = count.group() if count else "Unknown"
    
    return {
        "Brand": brand,
        "Product Type": product_type,
        "Size": size,
        "Count": count
    }

def process_multiple_images(image_paths):
    results = []
    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            extracted_text = process_image(img)
            product_info = extract_product_info(extracted_text)
            results.append({
                "Image": path,
                "Extracted Text": extracted_text,
                "Product Info": product_info
            })
        else:
            results.append({
                "Image": path,
                "Error": "Unable to read the image file."
            })
    return results

# Example usage
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
results = process_multiple_images(image_paths)

for result in results:
    print(f"\nImage: {result['Image']}")
    if 'Error' in result:
        print(f"Error: {result['Error']}")
    else:
        print(f"Extracted Text: {result['Extracted Text']}")
        print("Product Info:")
        for key, value in result['Product Info'].items():
            print(f"  {key}: {value}")