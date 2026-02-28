from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

MODEL_ID = "google/paligemma2-3b-pt-224"
ADAPTER_DIR = Path("./drowning_paligemma_adapter")
MODEL_LOADED = False
processor = None
model = None


def label_to_p_distress(label: str) -> float:
    label = label.strip().lower()
    if label == "drowning":
        return 0.95
    elif label == "swimming":
        return 0.05
    else:
        return 0.5


def _empty_result(error: str | None = None) -> dict:
    payload = {
        "detections": [],
        "threat_detected": False,
        "threat_count": 0,
    }
    if error:
        payload["error"] = error
    return payload


def _initialize_model(skip_adapter: bool = False) -> None:
    global MODEL_LOADED, processor, model

    try:
        print("Loading PaliGemma 2...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        adapter_config = ADAPTER_DIR / "adapter_config.json"
        if not skip_adapter and adapter_config.exists():
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
            print("Fine-tuned adapter loaded")
        else:
            print("No adapter found - running zero-shot")

        model.eval()
        MODEL_LOADED = True
        print("PaliGemma 2 ready.")
    except Exception as e:
        MODEL_LOADED = False
        processor = None
        model = None
        print("ERROR: Could not load PaliGemma 2")
        print("Check: huggingface-cli login and license accepted")
        print(f"Detail: {e}")


_initialize_model(skip_adapter=False)


def analyze_frame(image: Image.Image) -> dict:
    try:
        if not MODEL_LOADED:
            return _empty_result(error="Model not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = "<image> detect swimming ; drowning"
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0,
            )

        result = processor.decode(outputs[0], skip_special_tokens=True)

        pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([\w]+)"
        matches = re.findall(pattern, result)

        img_w, img_h = image.size
        detections: list[dict] = []

        for y1, x1, y2, x2, label in matches:
            x1_px = int(int(x1) / 1024 * img_w)
            y1_px = int(int(y1) / 1024 * img_h)
            x2_px = int(int(x2) / 1024 * img_w)
            y2_px = int(int(y2) / 1024 * img_h)

            norm_label = label.strip().lower()
            is_threat = norm_label == "drowning"

            detections.append(
                {
                    "label": norm_label,
                    "bbox": [x1_px, y1_px, x2_px, y2_px],
                    "is_threat": is_threat,
                    "p_distress": label_to_p_distress(norm_label),
                }
            )

        if not detections:
            return _empty_result()

        threat_count = sum(1 for d in detections if d["is_threat"])
        return {
            "detections": detections,
            "threat_detected": threat_count > 0,
            "threat_count": threat_count,
        }
    except Exception as e:
        print(f"analyze_frame error: {e}")
        return _empty_result(error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test image")
    parser.add_argument("--no-adapter", action="store_true", help="Skip adapter loading")
    args = parser.parse_args()

    if args.no_adapter:
        print("Reloading model without adapter...")
        _initialize_model(skip_adapter=True)

    img = Image.open(Path(args.test))
    print(f"Image size: {img.size}")
    print("Running inference...")

    result = analyze_frame(img)

    print(json.dumps(result, indent=2))
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Detection count: {result['threat_count']}")
