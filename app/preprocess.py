# app/preprocess.py
import io, math, numpy as np
from PIL import Image

WIDTH = 256

def bytes_to_image(byte_data: bytes, width: int = WIDTH) -> Image.Image:
    """실행파일 등 임의 바이트를 그레이스케일 이미지로 변환"""
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    H = math.ceil(len(arr) / width)
    if H == 0:
        H = 1
    pad_len = H*width - len(arr)
    if pad_len > 0:
        arr = np.pad(arr, (0, pad_len), constant_values=0)
    img = arr.reshape(H, width).astype(np.uint8)
    return Image.fromarray(img, mode="L")

def read_image_or_binary(data: bytes, is_binary_hint: bool = True) -> Image.Image:
    """
    실행파일(바이너리) 또는 PNG/JPEG 등을 받아 PIL Image로 반환.
    프론트에서 항상 exe를 올린다면 is_binary_hint=True 유지.
    """
    if not is_binary_hint:
        # 이미지로 시도
        try:
            return Image.open(io.BytesIO(data)).convert("L")
        except Exception:
            pass
    # 바이너리로 간주하여 변환
    return bytes_to_image(data, width=WIDTH)
