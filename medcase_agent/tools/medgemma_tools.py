"""Image inspection with optional local MedGemma or a compatible server."""
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests

MEDGEMMA_PIPE = None


def extract_image_panels(
    image_input: Any,
    separation_iters: int = 2,
    thresh_val: int = 240,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.90
) -> Dict[str, Dict[str, Any]]:
    """
    Splits a composite image into its sub-panels IN MEMORY.

    Args:
        image_input: The input image (Numpy array or PIL Image).
        separation_iters: Controls aggressive splitting of close panels.
        thresh_val: Pixel intensity threshold (0-255).
        min_area_ratio: Minimum area a panel must have to be kept.
        max_area_ratio: Maximum area a panel can have (filters out full image boundaries).

    Returns:
        dict: A dictionary where keys are panel names (e.g., 'panel_1') and values
              are dicts containing the cropped 'image' (np.ndarray) and its 'box' (x,y,w,h).
    """
    import cv2
    import numpy as np
    from PIL import Image

    # 1. Standardize Input to OpenCV format (Numpy BGR)
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError("Input must be a PIL Image or a numpy array.")

    img_h, img_w = image.shape[:2]
    total_area = img_h * img_w

    # 2. Image Preprocessing & Separation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if separation_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=separation_iters)

    # 3. Find and Filter Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if (total_area * min_area_ratio) < area < (total_area * max_area_ratio):
            pad = separation_iters
            x_pad, y_pad = max(0, x - pad), max(0, y - pad)
            w_pad, h_pad = min(img_w - x_pad, w + pad * 2), min(img_h - y_pad, h + pad * 2)
            panel_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # FALLBACK: If no panels pass, use the whole image
    if not panel_boxes:
        panel_boxes.append((0, 0, img_w, img_h))

    # 4. Sort roughly Top-to-Bottom, Left-to-Right
    panel_boxes = sorted(panel_boxes, key=lambda b: b[1])
    avg_height = sum([b[3] for b in panel_boxes]) / len(panel_boxes)
    y_tolerance = avg_height * 0.4

    rows = []
    current_row = [panel_boxes[0]]

    for box in panel_boxes[1:]:
        if abs(box[1] - current_row[-1][1]) <= y_tolerance:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    sorted_boxes = []
    for row in rows:
        sorted_row = sorted(row, key=lambda b: b[0])
        sorted_boxes.extend(sorted_row)

    # 5. Crop and Build Output Dictionary
    results = {}
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        chunk = image[y:y+h, x:x+w]
        results[f'panel_{i+1}'] = {
            'image': chunk,
            'box': (x, y, w, h)
        }

    return results

def get_medgemma_pipe():
    global MEDGEMMA_PIPE
    if MEDGEMMA_PIPE is None:
        if not os.getenv("MEDCASE_MEDGEMMA_MODEL"):
            raise RuntimeError("Set MEDCASE_MEDGEMMA_MODEL explicitly to enable local model inference")
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install medcase-agent[local-vision] or set MEDCASE_MEDGEMMA_BASE_URL") from exc
        MEDGEMMA_PIPE = pipeline(
            "image-text-to-text",
            model=os.environ["MEDCASE_MEDGEMMA_MODEL"],
            device=0 if torch.cuda.is_available() else -1,
        )
    return MEDGEMMA_PIPE


def analyze_radiology_image(
    image_reference_id: str,
    query: str = "Describe the visible medical image findings and any uncertainty.",
    execution_log: dict = None,
    case_data: dict = None,
    use_vllm: bool = None,
    vllm_url: str = None,
    vllm_model: str = None,
    **kwargs,
) -> str:
    """Inspect a registered image; heavy packages are loaded only for local inference."""
    images = (execution_log or {}).get("mapped_images", {})
    if image_reference_id not in images:
        return json.dumps({"error": "Unknown image reference ID"})
    root = Path((case_data or {}).get("metadata", {}).get("source_directory", ".")).resolve()
    image_path = (root / images[image_reference_id]).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        return json.dumps({"error": "Image is missing or outside the case image directory"})
    endpoint = vllm_url or os.getenv("MEDCASE_MEDGEMMA_BASE_URL")
    model = vllm_model or os.getenv("MEDCASE_MEDGEMMA_MODEL")
    if not model:
        return json.dumps({"error": "MedGemma is not configured. Set MEDCASE_MEDGEMMA_MODEL, plus MEDCASE_MEDGEMMA_BASE_URL for server inference or install medcase-agent[local-vision] for local inference."})
    use_remote = bool(endpoint) if use_vllm is None else use_vllm
    try:
        if use_remote:
            if not endpoint:
                return json.dumps({"error": "Set MEDCASE_MEDGEMMA_BASE_URL for server inference"})
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()}},
                ]}],
                "max_tokens": 1024,
            }
            headers = {"Content-Type": "application/json"}
            if os.getenv("OPENAI_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
            response = requests.post(endpoint.rstrip("/") + "/chat/completions", json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return json.dumps({"image": image_reference_id, "analysis": response.json()["choices"][0]["message"]["content"]})
        try:
            import cv2
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install medcase-agent[local-vision] or set MEDCASE_MEDGEMMA_BASE_URL") from exc
        original = cv2.imread(str(image_path))
        if original is None:
            return json.dumps({"error": "Could not decode registered image"})
        panels = extract_image_panels(original, separation_iters=3, thresh_val=240)
        pipe = get_medgemma_pipe()
        results = {}
        for name, panel in panels.items():
            image = Image.fromarray(cv2.cvtColor(panel["image"], cv2.COLOR_BGR2RGB))
            output = pipe(text=[{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": query}
            ]}], max_new_tokens=512)
            generated = output[0]["generated_text"]
            results[name] = generated[-1].get("content", "") if isinstance(generated, list) else generated
        return json.dumps(results)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Image analysis failed ({type(exc).__name__})"})
