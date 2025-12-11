import os
import cv2
import numpy as np
import time
from paddleocr import PPStructureV3
from evaluate_model import ModelEvaluator

def benchmark_ppstructure(image_path, gt_text, gt_boxes):
    """
    Runs PaddleOCR-VL (PP-StructureV3) on an image and evaluates it against Ground Truth.
    
    Args:
        image_path (str): Path to the image file.
        gt_text (str): Ground Truth text for the entire image.
        gt_boxes (list): Ground Truth bounding boxes list.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # 1. Initialize Model
    print("Initializing PP-StructureV3...")
    # use_table_recognition=False for speed, set to True if your task involves tables
    engine = PPStructureV3(use_table_recognition=False, lang='ch')
    
    # 2. Initialize Evaluator
    evaluator = ModelEvaluator()
    
    # 3. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Failed to load image.")
        return

    # 4. Measure Inference Speed
    print("Measuring inference speed...")
    def predict_wrapper(image_data):
        return engine.predict(image_data)
        
    speed_metrics = evaluator.measure_inference_speed(predict_wrapper, img, num_runs=5)
    print(f"Speed Metrics: {speed_metrics}")
    
    # 5. Get Prediction Result
    print("Running full inference...")
    result = engine.predict(img)[0]
    
    # 6. Parse Result
    # PP-Structure returns a list of regions. We need to aggregate text and boxes.
    pred_text_all = ""
    pred_boxes_all = []
    
    # Note: The structure of 'result' depends on the mode. 
    # For PPStructureV3, it usually contains 'structure_str', 'res', etc.
    # Here we iterate through regions.
    
    # Handle different return formats (list of dicts usually)
    regions = result if isinstance(result, list) else [result]
    
    for region in regions:
        # Check if region is a dict or object (handling potential variations)
        if isinstance(region, dict):
            res_type = region.get('type', '').lower()
            res_content = region.get('res', [])
        else:
            # Fallback if it's an object
            res_type = getattr(region, 'type', '').lower()
            res_content = getattr(region, 'res', [])

        if res_type == 'text':
            # 'res' is usually a list of dicts with 'text' and 'text_region'
            for line in res_content:
                text = line.get('text', '')
                box = line.get('text_region', [])
                
                pred_text_all += text
                if len(box) > 0:
                    pred_boxes_all.append(box)
                    
        elif res_type == 'table':
            # For tables, we might want to evaluate the HTML or cell text
            # Here we just append raw text if available
            if 'html' in region:
                pred_text_all += region['html'] # Or process further
                
    # 7. Calculate Accuracy Metrics
    print("\n--- Evaluation Results ---")
    
    # CER / WER
    cer = evaluator.calculate_cer(gt_text, pred_text_all)
    wer = evaluator.calculate_wer(gt_text, pred_text_all)
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
    
    # Detection Metrics
    if gt_boxes:
        det_metrics = evaluator.evaluate_detection(gt_boxes, pred_boxes_all)
        print(f"Detection Metrics: {det_metrics}")
    else:
        print("Skipping detection evaluation (no GT boxes provided).")

if __name__ == "__main__":
    # Example Usage
    # Replace these with your actual data
    # Using the sample.pdf converted to image for testing if jpg not present
    test_img = "data/sample_invoice.jpg" 
    
    # If sample.pdf exists, let's try to use the first page of it as test image
    pdf_path = "data/sample.pdf"
    if os.path.exists(pdf_path) and not os.path.exists(test_img):
        print(f"Converting first page of {pdf_path} to image for benchmarking...")
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap()
        test_img = "data/sample_page_0.jpg"
        pix.save(test_img)
        print(f"Saved {test_img}")
    
    # Mock Ground Truth (Replace with your label file reading logic)
    ground_truth_text = "发票代码: 0123456789\n开票日期: 2025年12月11日"
    ground_truth_boxes = [
        [[10, 10], [100, 10], [100, 30], [10, 30]], # Example box
        [[10, 40], [100, 40], [100, 60], [10, 60]]
    ]
    
    if not os.path.exists(test_img):
        print(f"Note: Test image '{test_img}' not found. Please place an image there to run.")
    else:
        benchmark_ppstructure(test_img, ground_truth_text, ground_truth_boxes)
