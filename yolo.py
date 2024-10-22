import cv2
import numpy as np
import pytesseract
from PIL import Image

def detect_products(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 30, 100)  # Lowered thresholds for more sensitivity
    
    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area = 1000  # Lowered minimum area
    max_area = frame.shape[0] * frame.shape[1] * 0.5  # Max 50% of frame
    product_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    return product_contours, edges, dilated

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect potential products
    product_contours, edges, dilated = detect_products(frame)
    
    # Create a copy of the frame for drawing
    debug_frame = frame.copy()
    
    for contour in product_contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle around the product
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Perform OCR on the detected region
        roi = frame[y:y+h, x:x+w]
        text = pytesseract.image_to_string(Image.fromarray(roi))
        
        # Clean up the text (remove newlines and extra spaces)
        text = ' '.join(text.split())
        
        # Display the OCR text
        cv2.putText(debug_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the number of detected contours
    cv2.putText(debug_frame, f"Detected: {len(product_contours)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display the resulting frames
    cv2.imshow('Product Detection', debug_frame)
    cv2.imshow('Edges', edges)
    cv2.imshow('Dilated Edges', dilated)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
