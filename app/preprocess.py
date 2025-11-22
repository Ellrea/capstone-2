import io
import math
import numpy as np
from PIL import Image

# 학습 시 사용했던 입력 크기와 동일하게 맞춘다
WIDTH = 256
TARGET_SIZE = 256  

def bytes_to_image(byte_data: bytes, width: int = WIDTH, target: int = TARGET_SIZE) -> Image.Image:
    """
    실행파일(raw bytes)을 grayscale 이미지로 변환한 뒤
    곧바로 target x target 크기로 축소하여 반환한다.
    (큰 이미지로 인해 Render 서버에서 메모리 폭발하던 문제 해결)
    """
    # 1) raw bytes → uint8 array
    arr = np.frombuffer(byte_data, dtype=np.uint8)

    # 2) (H, width)로 reshape
    H = math.ceil(len(arr) / width)
    if H <= 0:
        H = 1

    # 3) 패딩
    pad_len = H * width - len(arr)
    if pad_len > 0:
        arr = np.pad(arr, (0, pad_len), constant_values=0)

    img = arr.reshape(H, width).astype(np.uint8)

    # 4) L 채널 grayscale PIL 생성
    pil = Image.fromarray(img, mode="L")

    # 5)  핵심: 변환 직후 즉시 target × target 으로 축소 (메모리 절약)
    if (pil.width, pil.height) != (target, target):
        pil = pil.resize((target, target), resample=Image.BILINEAR)

    return pil


def read_image_or_binary(data: bytes, is_binary_hint: bool = True) -> Image.Image:
    """
    바이너리(EXE) 또는 JPEG/PNG 이미지 입력을 모두 처리.
    프론트에서 항상 exe를 올린다면 is_binary_hint=True 유지.
    """
    if not is_binary_hint:
        # 이미지로 시도
        try:
            img = Image.open(io.BytesIO(data)).convert("L")
            # 이미지도 바로 target 사이즈로
            return img.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.BILINEAR)
        except Exception:
            pass

    # 바이너리 또는 실패 시 bytes 변환
    return bytes_to_image(data, width=WIDTH, target=TARGET_SIZE)
