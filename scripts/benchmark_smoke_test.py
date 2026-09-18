import time
from pathlib import Path
from PIL import Image
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    load_mediscan_model,
    preprocess_image_for_inference,
    run_model_inference,
    compute_gradcam_explanation,
    get_ham10000_sample_catalog
)

device = "cpu"
print(f"Loading canonical model on {device}...")
model, meta = load_mediscan_model(device=device)

# Curated samples directory
curated_dir = PROJECT_ROOT / "app" / "assets" / "samples"
import json
with open(curated_dir / "samples_metadata.json", "r", encoding="utf-8") as f:
    curated_meta = json.load(f)

print(f"Testing on {len(curated_meta)} curated deployment samples...")
timings = []
for item in curated_meta:
    img_path = curated_dir / item["filename"]
    pil_img = Image.open(img_path)
    
    t0 = time.perf_counter()
    tensor, resized_rgb = preprocess_image_for_inference(pil_img)
    t_prep = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    pred_res = run_model_inference(model, tensor, device=device)
    t_infer = time.perf_counter() - t1
    
    t2 = time.perf_counter()
    cam_res = compute_gradcam_explanation(model, tensor, resized_rgb, alpha=0.45)
    t_cam = time.perf_counter() - t2
    
    total_t = t_prep + t_infer + t_cam
    timings.append({
        "class": item["dx"],
        "img_id": item["image_id"],
        "t_prep_ms": t_prep * 1000,
        "t_infer_ms": t_infer * 1000,
        "t_cam_ms": t_cam * 1000,
        "total_ms": total_t * 1000,
        "pred": pred_res["predicted_class"],
        "conf": pred_res["confidence"]
    })

print("\n--- ACTUAL MEASURED TIMINGS ON CPU ---")
for t in timings:
    conf_pct = t["conf"] * 100
    print(f"{t['img_id']} ({t['class']}): total={t['total_ms']:.1f}ms "
          f"(prep={t['t_prep_ms']:.1f}ms, infer={t['t_infer_ms']:.1f}ms, gradcam={t['t_cam_ms']:.1f}ms) "
          f"-> pred={t['pred']} ({conf_pct:.1f}%)")

avg_total = np.mean([t["total_ms"] for t in timings])
avg_infer = np.mean([t["t_infer_ms"] for t in timings])
avg_cam = np.mean([t["t_cam_ms"] for t in timings])
print(f"\nAverage Total End-to-End Latency: {avg_total:.1f} ms ({avg_total/1000:.3f} s)")
print(f"Average Preprocessing Latency: {np.mean([t['t_prep_ms'] for t in timings]):.1f} ms")
print(f"Average Model Forward Latency: {avg_infer:.1f} ms")
print(f"Average Grad-CAM Latency: {avg_cam:.1f} ms")
