import io
import math
import numpy as np
from PIL import Image

# 학습 시 사용했던 입력 크기와 동일하게 맞춘다
WIDTH = 256
TARGET_SIZE = 256  

def bytes_to_image(byte_data: bytes, width: int = WIDTH, target: int = TARGET_SIZE) -> Image.Image:
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

    # 5)  변환 직후 즉시 target × target 으로 축소 
    #if (pil.width, pil.height) != (target, target):
    #    pil = pil.resize((target, target), resample=Image.BILINEAR)

    return pil


def read_image_or_binary(data: bytes, is_binary_hint: bool = True) -> Image.Image:
    if not is_binary_hint:
        try:
            img = Image.open(io.BytesIO(data)).convert("L")
            w, h = img.size
            if w != WIDTH:
                new_h = int(h * (WIDTH / w))
                img = img.resize((WIDTH, new_h), resample=Image.BILINEAR)
            return img
        except Exception:
            pass

    # 실행파일은 bytes_to_image에서 이미 폭=WIDTH, 높이=H 로 만듦
    return bytes_to_image(data, width=WIDTH, target=TARGET_SIZE)
