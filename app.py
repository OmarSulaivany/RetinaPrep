import os
import base64
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from scipy.ndimage import label
from skimage.measure import regionprops
from PIL import Image
import io

# ====== Configuration ======
MODEL_NAME = "pamixsun/segformer_for_optic_disc_cup_segmentation"
FIXED_DIM = (1024, 1024) # Resize inputs to this before processing
app = Flask(__name__)

# ====== Load Model (Once at startup) ======
print("🔄 Loading AI Model... Please wait.")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
print("✅ Model Loaded!")

def to_base64(img_bgr):
    """Convert OpenCV image to Base64 string for HTML display"""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def get_best_bbox_and_stats(pred_mask, img_shape):
    """Finds the best optic disc region and calculates stats"""
    labeled_mask, _ = label(pred_mask > 0)
    regions = regionprops(labeled_mask)
    
    h, w = img_shape[:2]
    center_y, center_x = h / 2, w / 2
    
    # Defaults for "Bad" segmentation
    best_data = {
        "found": False,
        "bbox": None, # [minc, maxc, minr, maxr]
        "score": 0,
        "area": 0,
        "circularity": 0,
        "dist_factor": 0,
        "center_x": w//2, # Default to center if failed
        "center_y": h//2,
        "radius": 140
    }

    if not regions:
        return best_data

    best_score = -1

    for region in regions:
        area = region.area
        perimeter = region.perimeter if region.perimeter > 0 else 1
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)
        
        y, x = region.centroid
        distance = np.hypot(x - center_x, y - center_y)
        distance_factor = max(0, 1 - (distance / np.hypot(center_x, center_y)))
        
        score = area * circularity * distance_factor

        # Track best, but we keep stats even if low score
        if score > best_score:
            best_score = score
            minr, minc, maxr, maxc = region.bbox
            # Calculate radius based on bounding box largest side
            radius = max((maxc - minc), (maxr - minr)) / 2 
            
            best_data = {
                "found": True,
                "bbox": [int(minc), int(maxc), int(minr), int(maxr)],
                "score": float(score),
                "area": int(area),
                "circularity": float(circularity),
                "dist_factor": float(distance_factor),
                "center_x": int(x),
                "center_y": int(y),
                "radius": int(radius)
            }

    return best_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file'}), 400

    # 1. Read and Resize
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_h, original_w = img.shape[:2]
    
    # Resize to fixed square for consistent AI behavior
    img_resized = cv2.resize(img, FIXED_DIM, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 2. AI Inference
    inputs = processor(img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    upsampled_logits = nn.functional.interpolate(
        outputs.logits.cpu(), size=FIXED_DIM, mode="bilinear", align_corners=False
    )
    pred_mask = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)

    # 3. Create Visualizer Overlay (Green=Disc, Red=Cup)
    overlay = img_resized.copy()
    # Create semi-transparent masks
    zeros = np.zeros_like(overlay)
    mask_layer = zeros.copy()
    mask_layer[pred_mask == 1] = [0, 255, 0] # Green
    mask_layer[pred_mask == 2] = [0, 0, 255] # Red
    
    # Blend: 0.7 original + 0.3 mask
    has_mask = (pred_mask > 0)
    overlay[has_mask] = cv2.addWeighted(overlay[has_mask], 0.7, mask_layer[has_mask], 0.3, 0)

    # 4. Get Stats & Crop Coordinates
    stats = get_best_bbox_and_stats(pred_mask, img_resized.shape)

    return jsonify({
        'name': file.filename,
        'original_dims': f"{original_h}x{original_w}",
        'processed_dims': f"{FIXED_DIM[0]}x{FIXED_DIM[1]}",
        'img_original': to_base64(img_resized),
        'img_visual': to_base64(overlay),
        'stats': stats
    })

if __name__ == '__main__':
    port = 5000
    
    # 1. Print the clickable link for JupyterHub
    if 'JUPYTERHUB_SERVICE_PREFIX' in os.environ:
        # This gets '/user/omar/' automatically
        prefix = os.environ['JUPYTERHUB_SERVICE_PREFIX'] 
    else:
        # Fallback for local laptop
        print(f"\n🚀 App running locally: http://127.0.0.1:{port}/\n")

    # 2. Run Flask
    # host='0.0.0.0' is REQUIRED for JupyterHub to see the app
    app.run(debug=True, host='0.0.0.0', port=port)