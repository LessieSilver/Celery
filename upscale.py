import cv2
from cv2 import dnn_superres
import numpy as np

_scaler = None


def _get_scaler(model_path: str = 'EDSR_x2.pb'):
    global _scaler
    if _scaler is None:
        _scaler = dnn_superres.DnnSuperResImpl_create()
        _scaler.readModel(model_path)
        _scaler.setModel("edsr", 2)
    return _scaler


def upscale_bytes(image_bytes: bytes, model_path: str = 'EDSR_x2.pb') -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")

    scaler = _get_scaler(model_path)
    result = scaler.upsample(image)

    success, buffer = cv2.imencode('.png', result)
    if not success:
        raise ValueError("Failed to encode result image")
    return buffer.tobytes()


def example():
    with open('lama_300px.png', 'rb') as f:
        input_bytes = f.read()
    output_bytes = upscale_bytes(input_bytes)
    with open('lama_600px.png', 'wb') as f:
        f.write(output_bytes)


if __name__ == '__main__':
    example()