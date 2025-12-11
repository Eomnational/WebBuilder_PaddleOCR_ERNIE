import os
import fitz 
import cv2 
import numpy as np
from paddleocr import PaddleOCRVL 

def test_vl():
    pdf_path = "/home/ubuntu/heike/WebBuilder_PaddleOCR_ERNIE/data/sample.pdf"
    if not os.path.exists(pdf_path):
        print("PDF not found")
        return

    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            
        print("Initializing PaddleOCRVL...")
        engine = PaddleOCRVL()
        print("Running inference...")
        result = engine.predict(img_data)
        print("Result type:", type(result))
        print("Result:", result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    test_vl()
