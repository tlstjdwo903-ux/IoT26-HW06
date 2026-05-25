import io
import logging

import cv2
import easyocr
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_NAME = "keremberke/yolov8n-license-plate-detection"
CONFIDENCE_THRESHOLD = 0.4

# 서버 시작 시 모델 로드 (첫 호출 지연 방지)
logger.info("YOLO 모델 로딩 중...")
try:
    model = YOLO(MODEL_NAME)
    logger.info("번호판 전용 모델 로드 완료")
except Exception:
    logger.warning("번호판 모델 로드 실패 → yolov8n.pt 폴백")
    model = YOLO("yolov8n.pt")

logger.info("EasyOCR 초기화 중... (첫 실행 시 모델 다운로드)")
ocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
logger.info("EasyOCR 초기화 완료")


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def run_ocr(img: np.ndarray, bbox: list) -> tuple[str, float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # 크롭 영역을 약간 패딩
    h, w = img.shape[:2]
    pad = 4
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]

    results = ocr_reader.readtext(crop, detail=1)
    if not results:
        return "", 0.0

    # 신뢰도 가중 텍스트 합산
    texts = [r[1] for r in results]
    confidences = [r[2] for r in results]
    best_conf = max(confidences)
    combined_text = "".join(texts).replace(" ", "")
    return combined_text, round(best_conf, 4)


@app.route("/detect", methods=["POST"])
def detect():
    # multipart/form-data 또는 raw binary 모두 지원
    if request.content_type and "multipart" in request.content_type:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image 필드가 없습니다"}), 400
        image_bytes = request.files["image"].read()
    else:
        image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"success": False, "error": "이미지 데이터가 없습니다"}), 400

    try:
        img = decode_image(image_bytes)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    plates = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            bbox = box.xyxy[0].tolist()
            det_conf = round(float(box.conf[0]), 4)
            text, ocr_conf = run_ocr(img, bbox)
            plates.append({
                "text": text,
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_conf,
                "bbox": [int(v) for v in bbox],
            })

    logger.info("탐지 완료: %d개 번호판 발견", len(plates))
    return jsonify({"success": True, "plates": plates, "count": len(plates)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
