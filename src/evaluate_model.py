import numpy as np
import Levenshtein
from shapely.geometry import Polygon
import time

class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_cer(self, reference, hypothesis):
        """
        Calculate Character Error Rate (CER).
        CER = (S + D + I) / N
        S: Substitutions, D: Deletions, I: Insertions, N: Number of characters in reference
        """
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        dist = Levenshtein.distance(reference, hypothesis)
        return dist / len(reference)

    def calculate_wer(self, reference, hypothesis):
        """
        Calculate Word Error Rate (WER).
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
            
        # Construct a matrix for Levenshtein distance on words
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        Boxes are expected to be in format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        try:
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)

            if not poly1.is_valid or not poly2.is_valid:
                return 0.0

            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            if union == 0:
                return 0.0
            
            return intersection / union
        except Exception:
            return 0.0

    def calculate_f1_score(self, precision, recall):
        """
        Calculate F1 Score.
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_detection(self, gt_boxes, pred_boxes, iou_threshold=0.5):
        """
        Evaluate detection performance (Precision, Recall, F1) based on IoU.
        """
        matched_gt = set()
        tp = 0
        fp = 0
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        fn = len(gt_boxes) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def measure_inference_speed(self, model_predict_func, input_data, num_runs=10):
        """
        Measure average inference speed (FPS and Latency).
        """
        times = []
        # Warmup
        model_predict_func(input_data)
        
        for _ in range(num_runs):
            start_time = time.time()
            model_predict_func(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_latency = np.mean(times)
        fps = 1.0 / avg_latency if avg_latency > 0 else 0
        
        return {
            "avg_latency_sec": avg_latency,
            "fps": fps
        }

# Example Usage Mock
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # 1. CER & 2. WER Example
    ref_text = "PaddleOCR is an awesome tool for OCR."
    hyp_text = "PaddleOCR is a awesome tool for OCR."
    print(f"Reference: {ref_text}")
    print(f"Hypothesis: {hyp_text}")
    print(f"CER: {evaluator.calculate_cer(ref_text, hyp_text):.4f}")
    print(f"WER: {evaluator.calculate_wer(ref_text, hyp_text):.4f}")
    
    # 3. IoU & 4. Detection F1 Example
    # Box format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    gt_boxes = [
        [[10, 10], [50, 10], [50, 50], [10, 50]],
        [[60, 60], [100, 60], [100, 100], [60, 100]]
    ]
    pred_boxes = [
        [[12, 12], [48, 12], [48, 48], [12, 48]], # Good match
        [[200, 200], [250, 200], [250, 250], [200, 250]] # False positive
    ]
    
    det_metrics = evaluator.evaluate_detection(gt_boxes, pred_boxes)
    print(f"Detection Metrics: {det_metrics}")
    
    # 5. Inference Speed Example
    def mock_predict(data):
        time.sleep(0.1) # Simulate processing
        return "result"
        
    speed_metrics = evaluator.measure_inference_speed(mock_predict, "dummy_image")
    print(f"Speed Metrics: {speed_metrics}")
