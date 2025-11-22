import os, json, time
from typing import Dict, Any

import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import read_image_or_binary
from .model_loader import load_resnet_single


# =========================
# 환경 변수/경로
# =========================
CKPT_PATH = os.getenv("CKPT_PATH", "app/runs/resnet50_malimg.ckpt")
LABELS_PATH = os.getenv("LABELS_PATH", "app/labels.json")

IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "25"))
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
softmax = torch.nn.Softmax(dim=1)

# torch 성능/안정 옵션
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
torch.backends.cudnn.benchmark = True if DEVICE == "cuda" else False


#========================
# labels 로딩
# =========================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

idx_to_class: Dict[int, str] = {int(k): v for k, v in raw.items()}
num_classes = len(idx_to_class)


# =========================
#라벨별 malware type 매핑
# =========================
MALWARE_TYPE_MAP = {
    "Adialer.C": "Dialer",
    "Agent.FYI": "Backdoor",
    "Allaple.A": "Worm",
    "Allaple.L": "Worm",
    "Alueron.gen!J": "Worm",
    "Autorun.K": "Worm:AutoIT",
    "C2LOP.P": "Trojan",
    "C2LOP.gen!g": "Trojan",
    "Dialplatform.B": "Dialer",
    "Dontovo.A": "Trojan Downloader",
    "Fakerean": "Rogue",
    "Instantaccess": "Dialer",
    "Lolyda.AA1": "PWS",
    "Lolyda.AA2": "PWS",
    "Lolyda.AA3": "PWS",
    "Lolyda.AT": "PWS",
    "Malex.gen!J": "Trojan",
    "Obfuscator.AD": "Trojan Downloader",
    "Rbot!gen": "Backdoor",
    "Skintrim.N": "Trojan",
    "Swizzor.gen!E": "Trojan Downloader",
    "Swizzor.gen!I": "Trojan Downloader",
    "VB.AT": "Worm",
    "Wintrim.BX": "Trojan Downloader",
    "Yuner.A": "Worm",
    # "Benign" 은 악성 타입 없음 → None 처리
}


# ======================
# transform
# =========================
eval_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

def pil_to_tensor(pil):
    return eval_tfms(pil).unsqueeze(0)


# =========================
#서버 시작 시 1번만 고정 로드
# =========================
model = load_resnet_single(CKPT_PATH, num_classes=num_classes, device=DEVICE)
model.eval()


# =========================
# FastAPI
# =========================
app = FastAPI(
    title="PE Image Malware Classifier (Single ResNet)",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "ckpt_path": CKPT_PATH,
        "labels_path": LABELS_PATH,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    # -------------------------
    # 1) 파일 chunk read (메모리 피크 방지)
    # -------------------------
    data = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)  # 1MB
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (>{MAX_FILE_MB}MB)")
    data = bytes(data)

    # -------------------------
    # 2) PE bytes -> PIL (preprocess에서 256x256으로 축소됨)
    # -------------------------
    pil = read_image_or_binary(data, is_binary_hint=True)

    # -------------------------
    # 3) tensor -> inference
    # -------------------------
    x = pil_to_tensor(pil).to(DEVICE, non_blocking=True)

    t0 = time.time()
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
        else:
            logits = model(x)

        probs = softmax(logits)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_score = float(probs[pred_idx].item())

    #malware 여부 bool
    is_malware = (pred_label.lower() != "benign")

    #malware type
    malware_type = None if not is_malware else MALWARE_TYPE_MAP.get(pred_label, "Unknown")

    # top-k
    k = min(3, num_classes)
    topk = torch.topk(probs, k=k)
    top_list = [
        {
            "label": idx_to_class[int(i)],
            "index": int(i),
            "score": float(s),
            "malware_type": None
                if idx_to_class[int(i)].lower() == "benign"
                else MALWARE_TYPE_MAP.get(idx_to_class[int(i)], "Unknown")
        }
        for s, i in zip(topk.values.tolist(), topk.indices.tolist())
    ]

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "filename": file.filename,
        "is_malware": is_malware,
        "prediction": pred_label,
        "prediction_index": pred_idx,
        "malware_type": malware_type,
        "score": pred_score,
        "top_families": top_list,
        "latency_ms": latency_ms
    }
