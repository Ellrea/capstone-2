import os
import time
import json
from typing import List, Dict

import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import read_image_or_binary
from .model_loader import load_resnet_auto


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
softmax = torch.nn.Softmax(dim=1)


# ============================================================
# 안전한 float 환경변수 읽기 (Windows TEMP 충돌 방지)
# ============================================================
def _get_float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        print(f"[WARN] 환경변수 {name}='{val}' 은 float 변환 실패 → 기본값 {default} 사용")
        return default


# ============================================================
# 환경 변수
# ============================================================
# 1단계: binary (benign / malware) 모델 경로
BINARY_CKPT_PATH = os.getenv(
    "BINARY_CKPT_PATH",
    "app/runs/mal_cnn_binary/best_binary.ckpt"  # <- 너의 2클래스 모델 경로
)

# 2단계: multi (25 malware families) 모델 경로
MULTI_CKPT_PATH = os.getenv(
    "MULTI_CKPT_PATH",
    "app/runs/mal_cnn_25/best.ckpt"  # <- 새로 학습한 25클래스 모델 경로에 맞춰 수정
)

# 25개 패밀리 이름이 들어 있는 labels.json (지금 올려준 그 파일)
LABELS_PATH = os.getenv("LABELS_PATH", "app/labels.json")

# 1단계 binary 임계값: 이 값이 클수록 FP(정상→악성) 감소
TAU_BIN = _get_float_env("TAU_BIN", 0.95)          # 추천: 0.9~0.99에서 튜닝
LOGIT_TEMP_BIN = _get_float_env("LOGIT_TEMP_BIN", 1.0)

# 2단계 multi 온도 보정 (필요 없으면 1.0 유지)
LOGIT_TEMP_MULTI = _get_float_env("LOGIT_TEMP_MULTI", 1.0)

MAX_FILE_MB = _get_float_env("MAX_FILE_MB", 25.0)
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")


# ============================================================
# 이미지 전처리
# ============================================================
IMG_SIZE = 256
eval_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return eval_tfms(pil_img).unsqueeze(0)


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="Exe→Image Malware Classifier (2-stage, 25 families)",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 1단계: binary 모델 로드 (2-class)
# ============================================================
binary_model, binary_head_classes = load_resnet_auto(BINARY_CKPT_PATH, DEVICE)
if binary_head_classes != 2:
    raise RuntimeError(
        f"[binary] head_classes={binary_head_classes}, but expected 2. "
        f"→ BINARY_CKPT_PATH가 2클래스 모델인지 확인하세요."
    )


# ============================================================
# 2단계: multi 모델 로드 (25-class, 전부 악성 패밀리)
# ============================================================
multi_model, multi_head_classes = load_resnet_auto(MULTI_CKPT_PATH, DEVICE)

# 라벨 로드 (idx → 패밀리 이름)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
idx_to_class: Dict[int, str] = {int(k): v for k, v in raw.items()}

if multi_head_classes != len(idx_to_class):
    raise RuntimeError(
        f"[multi mismatch] 모델 출력 클래스 수={multi_head_classes} "
        f"vs labels.json 클래스 수={len(idx_to_class)}\n"
        f"→ MULTI_CKPT_PATH와 labels.json을 같은 데이터 기준으로 맞춰주세요."
    )


# ============================================================
# 헬스체크
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "binary_head_classes": binary_head_classes,
        "multi_head_classes": multi_head_classes,
        "num_families": len(idx_to_class),
        "tau_bin": TAU_BIN,
        "logit_temp_bin": LOGIT_TEMP_BIN,
        "logit_temp_multi": LOGIT_TEMP_MULTI,
    }


# ============================================================
# /analyze 엔드포인트
# ============================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 파일 크기 제한
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds limit ({MAX_FILE_MB} MB"
        )

    # 실행파일 → 이미지 변환
    pil = read_image_or_binary(data, is_binary_hint=True)
    x = pil_to_tensor(pil).to(DEVICE, non_blocking=True)

    t0 = time.time()

    # --------------------------------------------------------
    # 1단계: binary 모델로 benign / malware 판정
    # --------------------------------------------------------
    with torch.no_grad():
        logits_bin = binary_model(x)
        if LOGIT_TEMP_BIN != 1.0:
            logits_bin = logits_bin / LOGIT_TEMP_BIN
        probs_bin = softmax(logits_bin)[0]  # [P(benign), P(malware)]

    p_benign_bin = float(probs_bin[0])
    p_mal_bin = float(probs_bin[1])

    # binary 기준 악성 여부
    is_malware = p_mal_bin >= TAU_BIN
    latency_ms = int((time.time() - t0) * 1000)

    # --------------------------------------------------------
    # 1단계에서 benign으로 판단된 경우 → 바로 benign 반환
    # --------------------------------------------------------
    if not is_malware:
        return {
            "result": "benign",
            "label": "benign",           # 사람이 보기 쉬운 이름
            "label_index": 0,            # binary 입장에서 benign 인덱스
            "malware_family": None,      # 정상이라 악성 패밀리 이름 없음
            "score_malware": p_mal_bin,  # binary 기준 P(malware)
            "scores_binary": {
                "benign": p_benign_bin,
                "malware": p_mal_bin
            },
            "scores_multi": None,
            "top_families": None,
            "threshold_binary": TAU_BIN,
            "logit_temp_bin": LOGIT_TEMP_BIN,
            "logit_temp_multi": LOGIT_TEMP_MULTI,
            "meta": {
                "filename": file.filename,
                "latency_ms": latency_ms,
                "device": DEVICE,
                "stage": "binary_only"
            }
        }

    # --------------------------------------------------------
    # 1단계에서 malware로 판단된 경우 → 2단계 multi 모델로 family 예측
    # --------------------------------------------------------
    t1 = time.time()
    with torch.no_grad():
        logits_multi = multi_model(x)
        if LOGIT_TEMP_MULTI != 1.0:
            logits_multi = logits_multi / LOGIT_TEMP_MULTI
        probs_multi = softmax(logits_multi)[0]  # [25]

    # top-k 패밀리 후보
    k = min(3, multi_head_classes)
    topk = torch.topk(probs_multi, k=k)
    top_list = [
        {
            "label": idx_to_class[int(i)],
            "index": int(i),
            "score": float(s)
        }
        for s, i in zip(topk.values.tolist(), topk.indices.tolist())
    ]

    if len(top_list) > 0:
        family_name = top_list[0]["label"]
        family_index = top_list[0]["index"]
    else:
        family_name = "unknown"
        family_index = -1

    # 전체 패밀리 확률 맵(원하면 상위 몇 개만 쓰도록 프론트에서 처리 가능)
    scores_multi = {
        idx_to_class[i]: float(probs_multi[i])
        for i in range(multi_head_classes)
    }

    latency_ms_total = int((time.time() - t0) * 1000)

    return {
        "result": "malware",
        "label": family_name,           # 대표 패밀리 이름
        "label_index": family_index,
        "malware_family": family_name,  # 악성코드 이름(패밀리)
        "score_malware": p_mal_bin,     # 최종 악성 점수는 binary 모델 기준
        "scores_binary": {
            "benign": p_benign_bin,
            "malware": p_mal_bin
        },
        "scores_multi": scores_multi,
        "top_families": top_list,
        "threshold_binary": TAU_BIN,
        "logit_temp_bin": LOGIT_TEMP_BIN,
        "logit_temp_multi": LOGIT_TEMP_MULTI,
        "meta": {
            "filename": file.filename,
            "latency_ms": latency_ms_total,
            "device": DEVICE,
        }
    }
