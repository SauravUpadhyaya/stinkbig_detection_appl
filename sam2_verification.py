"""
SAM 2 Verification Module for Stinkbug Detection
Uses SAM 2 to double-check counts when approaching alert threshold
"""

import io
import numpy as np
from PIL import Image
import torch


class SAM2Verifier:
    """SAM 2 integration for verifying bug counts near threshold"""
    
    def __init__(self):
        """Initialize SAM 2 model (lazy loading)"""
        self._predictor = None
        self._device = None
        self._load_attempted = False
        self._load_error = None
        
    def _load_model(self):
        """Load SAM 2 model on first use"""
        if self._load_attempted:
            return
        
        self._load_attempted = True
        
        # Check Python version first
        import sys
        if sys.version_info < (3, 10):
            error_msg = f"SAM 2 requires Python 3.10+, but you're running {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            print(f"{error_msg}")
            print("   Falling back to YOLO-only counting (still accurate!)")
            print("   To use SAM 2: upgrade to Python 3.10+ or use YOLO-only mode")
            self._predictor = None
            self._load_error = error_msg
            return
            
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from pathlib import Path
            
            # Use smallest model for speed
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
            # Get absolute path to checkpoint file
            base_dir = Path(__file__).parent
            sam2_checkpoint = base_dir / "checkpoints" / "sam2_hiera_small.pt"
            
            # Check if checkpoint exists
            if not sam2_checkpoint.exists():
                error_msg = f"SAM 2 checkpoint not found at: {sam2_checkpoint}"
                print(f"{error_msg}")
                print("   Download it from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt")
                print(f"   Save to: {sam2_checkpoint}")
                self._predictor = None
                self._load_error = "Checkpoint file missing"
                return
            
            print(f"📦 Loading SAM 2 from {sam2_checkpoint}...")
            # Use config name (not path) - SAM 2 uses Hydra and looks in its own config directory
            sam2_model = build_sam2("sam2_hiera_s", str(sam2_checkpoint), device=device)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._device = device
            print(f"SAM 2 loaded successfully on {device}")
            self._load_error = None
            
        except ImportError as e:
            error_msg = f"SAM 2 package not installed: {e}"
            print(f"{error_msg}")
            print("   Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            self._predictor = None
            self._load_error = "Import error"
        except Exception as e:
            error_msg = f"SAM 2 loading failed: {e}"
            print(f"{error_msg}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("   Falling back to YOLO-only counting")
            self._predictor = None
            self._load_error = str(e)
    
    def verify_yolo_boxes(self, image_bytes: bytes, yolo_boxes: list) -> dict:
        """
        Use SAM 2 to verify YOLO detections and validate annotation quality
        
        Args:
            image_bytes: Raw image bytes
            yolo_boxes: List of [x1, y1, x2, y2] boxes from YOLO
            
        Returns:
            Dictionary with verification results and quality metrics:
            {
                'verified_count': int,
                'total_yolo_boxes': int,
                'avg_confidence': float,
                'quality_issues': list,
                'good_detections': list,
                'bad_detections': list,
                'annotation_quality': str (Excellent/Good/Fair/Poor)
            }
        """
        self._load_model()
        
        if self._predictor is None:
            # SAM 2 not available, trust YOLO
            print(f"⚠️ SAM2 verification skipped: predictor not loaded")
            return {
                'verified_count': len(yolo_boxes),
                'total_yolo_boxes': len(yolo_boxes),
                'avg_confidence': 1.0,
                'quality_issues': [],
                'good_detections': list(range(len(yolo_boxes))),
                'bad_detections': [],
                'annotation_quality': 'Unknown (SAM 2 not available)'
            }
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)
            img_height, img_width = image_np.shape[:2]
            
            self._predictor.set_image(image_np)
            
            verified_count = 0
            confidence_scores = []
            quality_issues = []
            good_detections = []
            bad_detections = []
        except Exception as e:
            print(f"❌ SAM2 verification failed during setup: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: trust YOLO
            return {
                'verified_count': len(yolo_boxes),
                'total_yolo_boxes': len(yolo_boxes),
                'avg_confidence': 1.0,
                'quality_issues': [],
                'good_detections': list(range(len(yolo_boxes))),
                'bad_detections': [],
                'annotation_quality': 'Unknown (SAM 2 verification failed)'
            }
        
        for idx, box in enumerate(yolo_boxes):
            try:
                input_box = np.array(box)
                
                # Get mask from box prompt
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                mask = masks[0]
                score = float(scores[0])
                mask_area = np.sum(mask)
                
                # Calculate box area for comparison
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                
                # SIMPLIFIED: For small objects, be very lenient
                is_small_object = box_area < 200
                
                # Very basic quality check for small objects
                if is_small_object:
                    # For small bugs: just check that SAM2 found SOMETHING
                    is_good = score > 0.4 and mask_area > 5
                else:
                    # For large objects: slightly more strict
                    is_good = score > 0.5 and mask_area > 20
                
                if is_good:
                    verified_count += 1
                    confidence_scores.append(score)
                    good_detections.append(idx)
                else:
                    bad_detections.append(idx)
                    quality_issues.append({
                        'box_index': idx,
                        'box': box,
                        'score': score,
                        'mask_area': int(mask_area),
                        'issues': [f"Low quality: score={score:.2f}, area={int(mask_area)}px"]
                    })
                    
            except Exception as e:
                print(f"❌ Error processing box {idx}: {e}")
                bad_detections.append(idx)
                quality_issues.append({
                    'box_index': idx,
                    'box': box,
                    'issues': [f"SAM 2 segmentation failed: {str(e)}"]
                })
                continue
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Determine overall annotation quality
        if len(yolo_boxes) == 0:
            quality = "No detections"
        else:
            good_ratio = verified_count / len(yolo_boxes)
            if good_ratio >= 0.9:
                quality = "Excellent"
            elif good_ratio >= 0.75:
                quality = "Good"
            elif good_ratio >= 0.5:
                quality = "Fair"
            else:
                quality = "Poor"
        
        return {
            'verified_count': verified_count,
            'total_yolo_boxes': len(yolo_boxes),
            'avg_confidence': avg_confidence,
            'quality_issues': quality_issues,
            'good_detections': good_detections,
            'bad_detections': bad_detections,
            'annotation_quality': quality
        }
    
    def recount_with_sam2(self, image_bytes: bytes) -> tuple[int, float]:
        """
        Re-analyze image completely with SAM 2 automatic segmentation
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            (count, average_confidence)
        """
        self._load_model()
        
        if self._predictor is None:
            return 0, 0.0
        
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)
            
            # Make SAM2 far more sensitive to tiny objects
            mask_generator = SAM2AutomaticMaskGenerator(
                model=self._predictor.model,
                points_per_side=48,          # Dense sampling to catch very small bugs
                pred_iou_thresh=0.3,         # Very lenient IoU threshold
                stability_score_thresh=0.6,  # More permissive stability
                crop_n_layers=2,             # Multi-scale crops to find small objects
            )
            
            masks = mask_generator.generate(image_np)
            
            # Filter masks by size and shape (more lenient for better detection)
            valid_masks = []
            rejected_reasons = []
            
            for m in masks:
                area = m['area']
                bbox = m['bbox']  # [x, y, width, height]
                width, height = bbox[2], bbox[3]
                confidence = m.get('predicted_iou', 0)
                
                # Check each filter
                reject_reason = None
                
                # Size filter: extremely lenient (4-25000 pixels)
                # Tiny stinkbugs can be <10 pixels in area when boxed tightly
                if not (4 < area < 25000):
                    reject_reason = f"Size out of range: {int(area)}px"

                # Aspect ratio filter: allow very thin/tall shapes (0.05-8.0)
                elif height > 0:
                    aspect_ratio = width / height
                    if not (0.05 < aspect_ratio < 8.0):
                        reject_reason = f"Bad aspect ratio: {aspect_ratio:.2f}"

                # Confidence filter (extremely lenient)
                elif confidence < 0.2:
                    reject_reason = f"Low confidence: {confidence:.2f}"

                # Exclude only ultra-thin noise (width or height < 1 pixel)
                elif width < 1 or height < 1:
                    reject_reason = "Too thin (likely noise)"
                
                else:
                    valid_masks.append(m)
                
                if reject_reason:
                    rejected_reasons.append((int(area), aspect_ratio if height > 0 else 0, confidence, reject_reason))
            
            if rejected_reasons:
                print(f"  Rejected {len(rejected_reasons)} masks:")
                for area, ratio, conf, reason in rejected_reasons[:5]:  # Show first 5
                    print(f"    - {reason}")
            
            # Remove very close duplicates (same stinkbug segmented twice)
            # Sort by area descending to keep largest masks
            valid_masks_sorted = sorted(valid_masks, key=lambda m: m['area'], reverse=True)
            deduped_masks = []
            
            for m1 in valid_masks_sorted:
                is_duplicate = False
                bbox1 = m1['bbox']
                center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
                area1 = m1['area']
                
                for m2 in deduped_masks:
                    bbox2 = m2['bbox']
                    center2 = (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)
                    area2 = m2['area']
                    
                    # If centers are very close (< 80 pixels) and sizes are similar, it's a duplicate
                    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2) ** 0.5
                    size_ratio = area1 / (area2 + 1e-6)
                    
                    # More strict duplicate detection: must be very similar size AND very close
                    if distance < 80 and 0.6 < size_ratio < 1.67:
                        is_duplicate = True
                        print(f"  Dedup: Removed duplicate (distance={distance:.0f}, size_ratio={size_ratio:.2f})")
                        break
                
                if not is_duplicate:
                    deduped_masks.append(m1)
            
            count = len(deduped_masks)
            avg_conf = np.mean([m['predicted_iou'] for m in deduped_masks]) if deduped_masks else 0.0
            
            print(f"🔍 SAM2 Independent: {len(masks)} total → {len(valid_masks)} candidates → {count} final stinkbugs (conf: {avg_conf:.2f})")
            
            return count, float(avg_conf)
            
        except Exception as e:
            print(f"SAM 2 automatic mode failed: {e}")
            return 0, 0.0
    
    def is_available(self) -> bool:
        """Check if SAM 2 is loaded and available"""
        self._load_model()
        return self._predictor is not None
    
    def get_status(self) -> dict:
        """Get SAM 2 availability status"""
        self._load_model()
        return {
            'available': self._predictor is not None,
            'device': self._device if self._predictor else None,
            'error': self._load_error
        }


# Global instance (lazy loaded)
_sam2_verifier = None

def get_sam2_verifier() -> SAM2Verifier:
    """Get or create global SAM 2 verifier instance"""
    global _sam2_verifier
    if _sam2_verifier is None:
        _sam2_verifier = SAM2Verifier()
    return _sam2_verifier
