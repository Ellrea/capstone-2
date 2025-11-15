import os
import time
import json
import gc
from typing import Dict

import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import read_image_or_binary
from .model_loader import load_resnet_auto


# ============================================================
# 디바이스 & 쓰레드 설정 (메모리/자원 절약)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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
    "app/runs/mal_cnn_binary/best_binary.ckpt"  # 2클래스 모델 경로
)

# 2단계: multi (25 malware families) 모델 경로
MULTI_CKPT_PATH = os.getenv(
    "MULTI_CKPT_PATH",
    "app/runs/mal_cnn_25/best.ckpt"  # 25클래스 모델 경로
)

# 25개 패밀리 이름이 들어 있는 labels.json
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
# ckpt에서 num_classes만 읽는 유틸 (메모리 절약용)
# ============================================================
def _get_ckpt_num_classes(ckpt_path: str) -> int:
    """ckpt의 fc.out_features(=num_classes)를 읽되, 모델은 만들지 않음."""
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["model_state"]
    num_classes = sd["fc.weight"].shape[0]
    # 메모리 해제
    del sd, blob
    gc.collect()
    return int(num_classes)


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
# 라벨 & 클래스 정보 로드 (가벼운 것들만 전역에 유지)
# ============================================================
# labels.json 로드 (idx → 패밀리 이름)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
idx_to_class: Dict[int, str] = {int(k): v for k, v in raw.items()}

multi_head_classes = _get_ckpt_num_classes(MULTI_CKPT_PATH)
if multi_head_classes != len(idx_to_class):
    raise RuntimeError(
        f"[multi mismatch] 모델 출력 클래스 수={multi_head_classes} "
        f"vs labels.json 클래스 수={len(idx_to_class)}\n"
        f"→ MULTI_CKPT_PATH와 labels.json을 같은 데이터 기준으로 맞춰주세요."
    )

binary_head_classes = _get_ckpt_num_classes(BINARY_CKPT_PATH)
if binary_head_classes != 2:
    raise RuntimeError(
        f"[binary] head_classes={binary_head_classes}, but expected 2. "
        f"→ BINARY_CKPT_PATH가 2클래스 모델인지 확인하세요."
    )


# ============================================================
# 요청마다 모델을 잠깐 로딩해서 쓰는 함수들 (메모리 절약의 핵심)
# ============================================================
def run_binary_head(x: torch.Tensor):
    """
    binary 모델을 요청 시점에 로딩 → 추론 → 바로 메모리에서 해제.
    512Mi 환경에서 두 모델이 동시에 상주하지 않도록 하기 위함.
    """
    model, num_classes = load_resnet_auto(BINARY_CKPT_PATH, DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    # 메모리 해제
    model.cpu()
    del model
    gc.collect()
    return logits, num_classes


def run_multi_head(x: torch.Tensor):
    """
    multi 모델을 요청 시점에 로딩 → 추론 → 바로 메모리에서 해제.
    binary 단계에서 악성으로 판정된 경우에만 호출.
    """
    model, num_classes = load_resnet_auto(MULTI_CKPT_PATH, DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    model.cpu()
    del model
    gc.collect()
    return logits, num_classes


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
    #   - 여기서만 binary 모델을 잠깐 로딩
    # --------------------------------------------------------
    logits_bin, _ = run_binary_head(x)
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
    # 1단계에서 malware로 판단된 경우 →
    #   2단계 multi 모델을 그때 로딩해서 family 예측
    # --------------------------------------------------------
    t1 = time.time()
    logits_multi, _ = run_multi_head(x)
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

    # 전체 패밀리 확률 맵
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
