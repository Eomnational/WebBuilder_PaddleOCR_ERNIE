import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from paddleocr import PPStructureV3
from PIL import Image, ImageDraw, ImageFont
import random

def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    
    # Load font
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    # Handle if result is a dict (extract list)
    # PPStructureV3 returns a dict with 'parsing_res_list' or similar
    regions = []
    if isinstance(result, dict):
        if 'parsing_res_list' in result:
            regions = result['parsing_res_list']
        elif 'res' in result:
            regions = result['res']
        else:
            # Fallback: maybe the dict itself is not what we expect, 
            # but let's try to see if it's iterable as a list of regions if it wasn't a dict
            pass
    elif isinstance(result, list):
        regions = result
    
    for region in regions:
        # Handle region being dict or object
        bbox = None
        label = 'unknown'
        
        if isinstance(region, dict):
            bbox = region.get('bbox') or region.get('text_region')
            label = region.get('type') or region.get('label') or 'unknown'
        else:
            # Assume object (e.g. StructureResult)
            bbox = getattr(region, 'bbox', None)
            if bbox is None:
                 bbox = getattr(region, 'text_region', None)
            label = getattr(region, 'type', None)
            if label is None:
                 label = getattr(region, 'label', 'unknown')
        
        if bbox:
            # Generate random color for each type
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw rectangle
            # bbox is [xmin, ymin, xmax, ymax]
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw label
            text_origin = (bbox[0], bbox[1] - 20 if bbox[1] > 20 else bbox[1])
            draw.text(text_origin, str(label), fill=color, font=font)
            
    return image

def visualize_pdf_layout(pdf_path, output_dir="output_vis"):
    """
    Process a PDF, run Layout Analysis, and save visualized images.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Initializing PP-StructureV3 for Layout Analysis...")
    # Enable layout analysis and table recognition
    # Note: layout=True is not a valid argument for PPStructureV3 in this version. 
    # It performs layout analysis by default.
    engine = PPStructureV3(use_table_recognition=True, lang='ch')

    doc = fitz.open(pdf_path)
    print(f"Processing {len(doc)} pages...")

    for page_num, page in enumerate(doc):
        print(f"Processing Page {page_num + 1}...")
        
        # Convert PDF page to image
        pix = page.get_pixmap(dpi=300) # High DPI for better detection
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            
        # Run Prediction
        result = engine.predict(img_data)[0]
        
        # Visualize
        # draw_structure_result returns a PIL Image
        im_show = draw_structure_result(Image.fromarray(img_data), result, font_path='./docs/fonts/simfang.ttf')
        
        # Save result
        output_filename = os.path.join(output_dir, f"page_{page_num + 1}_layout.jpg")
        im_show.save(output_filename)
        print(f"Saved visualization to {output_filename}")

        # Print detected regions for debugging
        print(f"--- Page {page_num + 1} Regions ---")
        
        regions = []
        if isinstance(result, dict):
            if 'parsing_res_list' in result:
                regions = result['parsing_res_list']
            elif 'res' in result:
                regions = result['res']
        elif isinstance(result, list):
            regions = result

        for region in regions:
            if isinstance(region, dict):
                r_type = region.get('type', 'unknown')
            else:
                r_type = getattr(region, 'type', 'unknown')
            print(f"  - Found {r_type}")

    print("Visualization complete.")

if __name__ == "__main__":
    # Ensure a font exists for drawing chinese characters, otherwise it might fail or show boxes
    # If simfang.ttf doesn't exist, you might need to provide a valid path or let it fallback
    # For this environment, we'll check if we need to download a font or use a system one.
    # But draw_structure_result usually needs a path.
    
    pdf_file = "/home/ubuntu/heike/WebBuilder_PaddleOCR_ERNIE/data/sample.pdf"
    visualize_pdf_layout(pdf_file)
